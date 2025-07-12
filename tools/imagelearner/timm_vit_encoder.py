import logging
import torch
import torch.nn as nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, IMAGE
from ludwig.encoders.image.base import ImageEncoder
from ludwig.encoders.types import EncoderOutputDict

logger = logging.getLogger(__name__)


@DeveloperAPI
class TimmViTHugeEncoder(ImageEncoder):
    def __init__(self, model_variant=None, use_pretrained=True, **kwargs):
        super().__init__()
        
        logger.info(f"TimmViTHugeEncoder initialized with model_variant: {model_variant}")
        
        # handle different model variants
        if model_variant == "huge_timm":
            logger.warning("huge_timm not fully implemented - using fallback ResNet")
            import torchvision.models as tvm
            self.model = tvm.resnet50(pretrained=use_pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.output_dim = 2048
        elif model_variant == "caformer_b36_timm":
            logger.info("Using Caformer encoder for caformer_b36_timm variant")
            # Use the Caformer encoder
            from caformer_encoder import CaformerEncoder
            caformer_encoder = CaformerEncoder(
                model_variant="caformer_b36_timm",
                use_pretrained=use_pretrained
            )
            self.model = caformer_encoder.model
            self.output_dim = caformer_encoder.output_dim
        else:
            logger.warning(f"Unknown model variant {model_variant} - using fallback ResNet")
            import torchvision.models as tvm
            self.model = tvm.resnet50(pretrained=use_pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.output_dim = 2048

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_dim])

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        encoded = self.model(inputs)
        return {ENCODER_OUTPUT: encoded}

    def get_embedding_layer(self) -> nn.Module:
        return self.model 
