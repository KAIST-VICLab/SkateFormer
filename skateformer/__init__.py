"""SkateFormer — Hugging Face Hub packaging of the ECCV 2024 model.

>>> from skateformer import SkateFormer
>>> model = SkateFormer.from_pretrained("KAIST-VICLab/SkateFormer-ntu60-xsub-joint").eval()
"""

from .modeling_skateformer import SkateFormer, SkateFormer_
from .labels import LABEL_SETS, id2label
from . import preprocessing

__all__ = ["SkateFormer", "SkateFormer_", "LABEL_SETS", "id2label", "preprocessing"]
__version__ = "0.1.0"
