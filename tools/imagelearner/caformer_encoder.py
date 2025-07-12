import logging
import os
import sys
from typing import Optional, Type, Union

import torch
import torch.nn as nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, IMAGE
from ludwig.encoders.image.base import ImageEncoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.schema.encoders.base import BaseEncoderConfig

logger = logging.getLogger(__name__)


@DeveloperAPI
class CaformerEncoder(ImageEncoder):
    """Caformer encoder using timm models."""
    
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

        # create timm model
        if model_variant == "caformer_b36_timm":
            model_name = "caformer_b36.sail_in22k_ft_in1k"
        else:
            raise ValueError(f"Unsupported model variant: {model_variant}")

        try:
            import timm
            
            logger.info(f"Loading timm model: {model_name}")
            self.model = timm.create_model(
                model_name,
                pretrained=use_pretrained,
                num_classes=0,  # remove classifier head
                global_pool='avg'
            )
            
            # the output dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                output = self.model(dummy_input)
                self.output_dim = output.shape[1]
                
            logger.info(f"CaformerEncoder initialized with timm model {model_name}, output dimension: {self.output_dim}")
            
        except ImportError:
            logger.error(
                "The timm library is not installed. "
                "To use the timm pretrained models as a Ludwig image "
                "encoders, please run pip install timm."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to create timm model {model_name}: {e}")
            raise

        # trainable parameter
        for param in self.model.parameters():
            param.requires_grad = trainable

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_dim])

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        """
        Forward pass through the Caformer model.
        
        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dictionary containing the encoded features
        """
        # ensure input is in the correct format
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        # forward pass through the model
        encoded = self.model(inputs)
        
        return {ENCODER_OUTPUT: encoded}

    def get_embedding_layer(self) -> nn.Module:
        return self.model


# register the encoder
@register_encoder("caformer", IMAGE)
class CaformerEncoderConfig(BaseEncoderConfig):
    type: str = "caformer"
    model_variant: str = "caformer_b36_timm"
    use_pretrained: bool = True
    trainable: bool = True 
