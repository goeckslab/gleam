import logging
import os
from typing import Optional, Type, Union, Literal

import torch
import torch.nn as nn
import timm

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, IMAGE
from ludwig.encoders.image.base import ImageEncoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.utils.image_utils import register_torchvision_model_variants, torchvision_model_registry, TVModelVariant

logger = logging.getLogger(__name__)


@DeveloperAPI
class CaformerEncoder(ImageEncoder):
    def __init__(
        self,
        model_variant: str = "caformer_b36_timm",
        use_pretrained: bool = True,
        saved_weights_in_checkpoint: bool = False,
        model_cache_dir: Optional[str] = None,
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__()

        logger.debug(f"Initializing CaformerEncoder with variant: {model_variant}")
        
        self.model_variant = model_variant
        self.use_pretrained = use_pretrained
        self.model_cache_dir = model_cache_dir
        self.trainable = trainable

        if model_variant == "caformer_b36_timm":
            model_name = "caformer_b36.sail_in22k_ft_in1k"
        elif model_variant == "caformer_s18_timm":
            model_name = "caformer_s18.sail_in22k_ft_in1k"
        else:
            raise ValueError(f"Unsupported model variant: {model_variant}")

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=use_pretrained,
                num_classes=0,
                global_pool='avg'
            )
            
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                output = self.model(dummy_input)
                self.output_dim = output.shape[1]
                
            logger.info(f"CaformerEncoder initialized with output dimension: {self.output_dim}")
            
        except Exception as e:
            logger.error(f"Failed to create timm model {model_name}: {e}")
            raise

        for param in self.model.parameters():
            param.requires_grad = trainable

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_dim])

    @classmethod
    def get_schema_cls(cls) -> Type[BaseEncoderConfig]:
        return CaformerEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([3, 224, 224])

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        # convert grayscale to RGB if needed
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        encoded = self.model(inputs)
        return {ENCODER_OUTPUT: encoded}

    def get_embedding_layer(self) -> nn.Module:
        return self.model


@register_encoder("caformer", IMAGE)
class CaformerEncoderConfig(BaseEncoderConfig):
    type: str = "caformer"
    model_variant: Literal["caformer_b36_timm", "caformer_s18_timm"] = "caformer_b36_timm"
    use_pretrained: bool = True
    trainable: bool = True 
