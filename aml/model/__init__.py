from .autoencoder_lightning import AEModel
from .mlp_lightning import MLPModel
from .transformer_encoder import TransformerEncoderModel
from .resnet1d import ResNet1DModel
from .base_model import BaseLightningModule
from .kdtree import KDAttributeTree

import warnings
import logging

## pytorch lightning warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings(
    "ignore", ".*You defined a `validation_step` but have no `val_dataloader`.*"
)
warnings.filterwarnings(
    "ignore", ".*GPU available but not used.*"
)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


__all__ = [
    "AEModel",
    "MLPModel",
    "TransformerEncoderModel",
    "ResNet1DModel",
    "BaseLightningModule",
    "KDAttributeTree",
]
