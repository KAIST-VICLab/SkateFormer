from typing import Type, Tuple, Optional, Set, List, Union

import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath, create_act_layer, get_norm_act_layer, create_conv2d

''' Partition and Reverse '''


def type_1_partition(input, partition_size):  # partition_size = [N, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_1_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_2_partition(input, partition_size):  # partition_size = [N, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_2_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)
    return output


def type_3_partition(input, partition_size):  # partition_size = [M, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_3_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_4_partition(input, partition_size):  # partition_size = [M, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_4_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)
    return output


''' 1D relative positional bias: B_{h}^{t} '''


def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)


''' MSA '''


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=32, partition_size=(1, 1), attn_drop=0., rel=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.rel_type = rel_type
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads ** -0.5
        self.attn_area = partition_size[0] * partition_size[1]
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rel = rel

        if self.rel:
            if self.rel_type == 'type_1' or self.rel_type == 'type_3':
                self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size[0] - 1), num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)
                self.ones = torch.ones(partition_size[1], partition_size[1], num_heads)
            elif self.rel_type == 'type_2' or self.rel_type == 'type_4':
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1], num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(self):
        if self.rel_type == 'type_1' or self.rel_type == 'type_3':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], -1)
            relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(3).repeat(1, self.partition_size[1], 1, self.partition_size[1], 1, 1).view(self.attn_area, self.attn_area, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)
        elif self.rel_type == 'type_2' or self.rel_type == 'type_4':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.partition_size[0], self.partition_size[0], self.partition_size[1], self.partition_size[1], -1)
            relative_position_bias = relative_position_bias.permute(0, 2, 1, 3, 4).contiguous().view(self.attn_area, self.attn_area, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)

    def forward(self, input):
        B_, N, C = input.shape
        qkv = input.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rel:
            attn = attn + self._get_relative_positional_bias()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        return output


''' SkateFormer Block '''


class SkateFormerBlock(nn.Module):
    def __init__(self, in_channels, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SkateFormerBlock, self).__init__()
        self.type_1_size = type_1_size
        self.type_2_size = type_2_size
        self.type_3_size = type_3_size
        self.type_4_size = type_4_size
        self.partition_function = [type_1_partition, type_2_partition, type_3_partition, type_4_partition]
        self.reverse_function = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse]
        self.partition_size = [type_1_size, type_2_size, type_3_size, type_4_size]
        self.rel_type = ['type_1', 'type_2', 'type_3', 'type_4']

        self.norm_1 = norm_layer(in_channels)
        self.mapping = nn.Linear(in_features=in_channels, out_features=2 * in_channels, bias=True)
        self.gconv = nn.Parameter(torch.zeros(num_heads // (2 * 2), num_points, num_points))
        trunc_normal_(self.gconv, std=.02)
        self.tconv = nn.Conv2d(in_channels // (2 * 2), in_channels // (2 * 2), kernel_size=(kernel_size, 1),
                               padding=((kernel_size - 1) // 2, 0), groups=num_heads // (2 * 2))

        # Attention layers
        attention = []
        for i in range(len(self.partition_function)):
            attention.append(
                MultiHeadSelfAttention(in_channels=in_channels // (len(self.partition_function) * 2),
                                       rel_type=self.rel_type[i],
                                       num_heads=num_heads // (len(self.partition_function) * 2),
                                       partition_size=self.partition_size[i], attn_drop=attn_drop, rel=rel))
        self.attention = nn.ModuleList(attention)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(mlp_ratio * in_channels),
                       act_layer=act_layer, drop=drop)

    def forward(self, input):
        B, C, T, V = input.shape

        # Partition
        input = input.permute(0, 2, 3, 1).contiguous()
        skip = input

        f = self.mapping(self.norm_1(input)).permute(0, 3, 1, 2).contiguous()

        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)
        y = []

        # G-Conv
        split_f_conv = torch.chunk(f_conv, 2, dim=1)
        y_gconv = []
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)
        for i in range(self.gconv.shape[0]):
            z = torch.einsum('n c t u, v u -> n c t v', split_f_gconv[i], self.gconv[i])
            y_gconv.append(z)
        y.append(torch.cat(y_gconv, dim=1))  # N C T V

        # T-Conv
        y.append(self.tconv(split_f_conv[1]))

        # Skate-MSA
        split_f_attn = torch.chunk(f_attn, len(self.partition_function), dim=1)

        for i in range(len(self.partition_function)):
            C = split_f_attn[i].shape[1]
            input_partitioned = self.partition_function[i](split_f_attn[i], self.partition_size[i])
            input_partitioned = input_partitioned.view(-1, self.partition_size[i][0] * self.partition_size[i][1], C)
            y.append(self.reverse_function[i](self.attention[i](input_partitioned), (T, V), self.partition_size[i]))

        output = self.proj(torch.cat(y, dim=1).permute(0, 2, 3, 1).contiguous())
        output = self.proj_drop(output)
        output = skip + self.drop_path(output)

        # Feed Forward
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


''' Downsampling '''


class PatchMergingTconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1),
                                   dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x


''' SkateFormer Block with Downsampling '''


class SkateFormerBlockDS(nn.Module):
    def __init__(
            self, in_channels, out_channels, num_points=50, kernel_size=7, downscale=False, num_heads=32,
            type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
            attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm):
        super(SkateFormerBlockDS, self).__init__()

        if downscale:
            self.downsample = PatchMergingTconv(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.downsample = None

        self.transformer = SkateFormerBlock(
            in_channels=out_channels,
            num_points=num_points,
            kernel_size=kernel_size,
            num_heads=num_heads,
            type_1_size=type_1_size,
            type_2_size=type_2_size,
            type_3_size=type_3_size,
            type_4_size=type_4_size,
            attn_drop=attn_drop,
            drop=drop,
            rel=rel,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer,
        )

    def forward(self, input):
        if self.downsample is not None:
            output = self.transformer(self.downsample(input))
        else:
            output = self.transformer(input)
        return output


''' SkateFormer Stage '''


class SkateFormerStage(nn.Module):
    def __init__(
            self, depth, in_channels, out_channels, first_depth=False,
            num_points=50, kernel_size=7, num_heads=32,
            type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
            attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm):
        super(SkateFormerStage, self).__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                SkateFormerBlockDS(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    num_points=num_points,
                    kernel_size=kernel_size,
                    downscale=((index == 0) & ~first_depth),
                    num_heads=num_heads,
                    type_1_size=type_1_size,
                    type_2_size=type_2_size,
                    type_3_size=type_3_size,
                    type_4_size=type_4_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    rel=rel,
                    drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        output = input
        for block in self.blocks:
            output = block(output)
        return output


''' SkateFormer '''


class SkateFormer(nn.Module):
    def __init__(self, in_channels=3, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), num_classes=60,
                 embed_dim=64, num_people=2, num_frames=64, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., head_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm, index_t=False, global_pool='avg'):

        super(SkateFormer, self).__init__()

        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        self.num_classes: int = num_classes
        self.head_drop = head_drop
        self.index_t = index_t
        self.embed_dim = embed_dim

        if self.head_drop != 0:
            self.dropout = nn.Dropout(p=self.head_drop)
        else:
            self.dropout = None

        stem = []
        stem.append(nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0)))
        stem.append(act_layer())
        stem.append(
            nn.Conv2d(in_channels=2 * in_channels, out_channels=3 * in_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)))
        stem.append(act_layer())
        stem.append(nn.Conv2d(in_channels=3 * in_channels, out_channels=embed_dim, kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0)))
        self.stem = nn.ModuleList(stem)

        if self.index_t:
            self.joint_person_embedding = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
            trunc_normal_(self.joint_person_embedding, std=.02)
        else:
            self.joint_person_temporal_embedding = nn.Parameter(
                torch.zeros(1, embed_dim, num_frames, num_points * num_people))
            trunc_normal_(self.joint_person_temporal_embedding, std=.02)

        # Init blocks
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            stages.append(
                SkateFormerStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    first_depth=index == 0,
                    num_points=num_points * num_people,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    type_1_size=type_1_size,
                    type_2_size=type_2_size,
                    type_3_size=type_3_size,
                    type_4_size=type_4_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    rel=rel,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer
                )
            )
        self.stages = nn.ModuleList(stages)
        self.global_pool: str = global_pool
        self.head = nn.Linear(channels[-1], num_classes)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes: int = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, input):
        output = input
        for stage in self.stages:
            output = stage(output)
        return output

    def forward_head(self, input, pre_logits=False):
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        if self.dropout is not None:
            input = self.dropout(input)
        return input if pre_logits else self.head(input)

    def forward(self, input, index_t):
        B, C, T, V, M = input.shape

        output = input.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M * V]
        for layer in self.stem:
            output = layer(output)
        if self.index_t:
            te = torch.zeros(B, T, self.embed_dim).to(output.device)  # B, T, C
            div_term = torch.exp(
                (torch.arange(0, self.embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.embed_dim))).to(
                output.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum('b t c, c v -> b c t v', te, self.joint_person_embedding)
        else:
            output = output + self.joint_person_temporal_embedding
        output = self.forward_features(output)
        output = self.forward_head(output)
        return output


def SkateFormer_(**kwargs):
    return SkateFormer(
        depths=(2, 2, 2, 2),
        channels=(96, 192, 192, 192),
        embed_dim=96,
        **kwargs
    )
