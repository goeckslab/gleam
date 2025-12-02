import logging
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_PREFIXES = (
    'identityformer_',
    'randformer_',
    'poolformerv2_',
    'convformer_',
    'caformer_',
)

try:
    from metaformer_models import default_cfgs as META_DEFAULT_CFGS
    META_MODELS_AVAILABLE = True
    logger.info("MetaFormer models imported successfully")
except Exception as e:
    META_MODELS_AVAILABLE = False
    logger.warning(f"MetaFormer models not available: {e}")


def _resolve_metaformer_ctor(model_name: str):
    # Prefer getattr to avoid importing every factory explicitly
    try:
        # Import the module itself for dynamic access
        import metaformer_models
        _factories = metaformer_models.__dict__
        if model_name in _factories and callable(_factories[model_name]):
            return _factories[model_name]
    except Exception:
        pass
    return None


class MetaFormerStackedCNN(nn.Module):
    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        num_channels: int = 3,
        output_size: int = 128,
        custom_model: str = "identityformer_s12",
        use_pretrained: bool = True,
        trainable: bool = True,
        conv_layers: Optional[List[Dict]] = None,
        num_conv_layers: Optional[int] = None,
        conv_activation: str = "relu",
        conv_dropout: float = 0.0,
        conv_norm: Optional[str] = None,
        conv_use_bias: bool = True,
        fc_layers: Optional[List[Dict]] = None,
        num_fc_layers: int = 1,
        fc_activation: str = "relu",
        fc_dropout: float = 0.0,
        fc_norm: Optional[str] = None,
        fc_use_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        logger.info("MetaFormerStackedCNN encoder instantiated")
        logger.info(f"Using MetaFormer model: {custom_model}")

        try:
            height = int(height)
            width = int(width)
            num_channels = int(num_channels)
        except (TypeError, ValueError) as exc:
            raise ValueError("MetaFormerStackedCNN requires integer height, width, and num_channels.") from exc

        if height <= 0 or width <= 0:
            raise ValueError(f"MetaFormerStackedCNN received non-positive dimensions: {height}x{width}.")
        if num_channels <= 0:
            raise ValueError(f"MetaFormerStackedCNN requires num_channels > 0, received {num_channels}.")

        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.output_size = output_size
        self.custom_model = custom_model
        self.use_pretrained = use_pretrained
        self.trainable = trainable

        cfg = META_DEFAULT_CFGS.get(custom_model, {})
        input_size = cfg.get('input_size', (3, 224, 224))
        if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
            expected_channels, expected_height, expected_width = input_size
        else:
            expected_channels, expected_height, expected_width = 3, 224, 224

        self.expected_channels = expected_channels
        self.expected_height = expected_height
        self.expected_width = expected_width

        logger.info(f"Initializing MetaFormerStackedCNN with model: {custom_model}")
        logger.info(
            "Input: %sx%sx%s -> Output: %s (expected backbone size: %sx%s)",
            num_channels,
            height,
            width,
            output_size,
            self.expected_height,
            self.expected_width,
        )

        self.channel_adapter: Optional[nn.Conv2d] = None
        if num_channels != self.expected_channels:
            self.channel_adapter = nn.Conv2d(
                num_channels, self.expected_channels, kernel_size=1, stride=1, padding=0
            )
            logger.info(
                "Added channel adapter: %s -> %s channels",
                num_channels,
                self.expected_channels,
            )

        self.size_adapter: Optional[nn.Module] = None
        if height != self.expected_height or width != self.expected_width:
            self.size_adapter = nn.AdaptiveAvgPool2d((height, width))
            logger.info(
                "Configured size adapter to requested input: %sx%s",
                height,
                width,
            )
        self.backbone_adapter: Optional[nn.Module] = None

        self.backbone = self._load_metaformer_backbone()
        self.feature_dim = self._get_feature_dim()

        self.fc_layers = self._create_fc_layers(
            input_dim=self.feature_dim,
            output_dim=output_size,
            num_layers=num_fc_layers,
            activation=fc_activation,
            dropout=fc_dropout,
            norm=fc_norm,
            use_bias=fc_use_bias,
            fc_layers_config=fc_layers,
        )

        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("MetaFormer backbone frozen (trainable=False)")

        logger.info("MetaFormerStackedCNN initialized successfully")

    def _load_metaformer_backbone(self):
        if not META_MODELS_AVAILABLE:
            raise ImportError("MetaFormer models are not available")

        ctor = _resolve_metaformer_ctor(self.custom_model)
        if ctor is None:
            raise ValueError(f"Unknown MetaFormer model: {self.custom_model}")

        logger.info("MetaFormer backbone requested: %s, use_pretrained=%s", self.custom_model, self.use_pretrained)
        cfg = META_DEFAULT_CFGS.get(self.custom_model, {})
        logger.info("MetaFormer cfg present=%s", bool(cfg))
        if not cfg:
            logger.warning("MetaFormer config missing for %s; will fall back to random initialization", self.custom_model)
        weights_url = cfg.get('url') if isinstance(cfg, dict) else None
        logger.info("MetaFormer weights_url=%s", weights_url)
        # track loading
        self._pretrained_loaded = False
        self._loaded_weights_url: Optional[str] = None
        if self.use_pretrained and weights_url:
            print(f"LOADING MetaFormer pretrained weights from: {weights_url}")
            logger.info(f"Loading pretrained weights from: {weights_url}")
        elif self.use_pretrained and not weights_url:
            logger.warning("MetaFormer: no pretrained URL found for %s; continuing with random weights", self.custom_model)
            self.use_pretrained = False
        # Ensure we log whenever the factories call torch.hub.load_state_dict_from_url
        orig_loader = getattr(torch.hub, 'load_state_dict_from_url', None)

        def _wrapped_loader(url, *args, **kwargs):
            print(f"DOWNLOADING weights from: {url}")
            logger.info(f"DOWNLOADING weights from: {url}")
            self._pretrained_loaded = True
            self._loaded_weights_url = url
            result = orig_loader(url, *args, **kwargs)
            print(f"WEIGHTS DOWNLOADED successfully from: {url}")
            return result
        try:
            if self.use_pretrained and orig_loader is not None:
                torch.hub.load_state_dict_from_url = _wrapped_loader  # type: ignore[attr-defined]
            print(f"CREATING MetaFormer model: {self.custom_model} (pretrained={self.use_pretrained})")
            try:
                model = ctor(pretrained=self.use_pretrained, num_classes=1000)
                print(f"MetaFormer model CREATED: {self.custom_model}")
            except Exception as model_error:
                if self.use_pretrained:
                    print(f"⚠ Warning: Failed to load {self.custom_model} with pretrained weights: {model_error}")
                    print("Attempting to load without pretrained weights as fallback...")
                    logger.warning(f"Failed to load {self.custom_model} with pretrained weights: {model_error}")
                    model = ctor(pretrained=False, num_classes=1000)
                    print(f"✓ Successfully loaded {self.custom_model} without pretrained weights")
                    self.use_pretrained = False  # Update state to reflect actual loading
                else:
                    raise model_error
        finally:
            if orig_loader is not None:
                torch.hub.load_state_dict_from_url = orig_loader  # type: ignore[attr-defined]
        self._metaformer_weights_url = weights_url
        if self.use_pretrained:
            if self._pretrained_loaded:
                print(f"MetaFormer: pretrained weights loaded from {self._loaded_weights_url}")
                logger.info(f"MetaFormer: pretrained weights loaded from {self._loaded_weights_url}")
            else:
                # Warn but don't fail - weights may have failed to load but model creation succeeded
                print("⚠ Warning: MetaFormer pretrained weights were requested but not confirmed as loaded")
                logger.warning("MetaFormer: pretrained weights were requested but not confirmed as loaded")
        else:
            print(f"MetaFormer: using randomly initialized weights for {self.custom_model}")
            logger.info(f"MetaFormer: using randomly initialized weights for {self.custom_model}")
        logger.info(f"Loaded MetaFormer backbone: {self.custom_model} (pretrained={self.use_pretrained})")
        return model

    def _get_feature_dim(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone.forward_features(dummy_input)
            feature_dim = features.shape[-1]
        logger.info(f"MetaFormer feature dimension: {feature_dim}")
        return feature_dim

    def _create_fc_layers(self, input_dim, output_dim, num_layers, activation, dropout, norm, use_bias, fc_layers_config):
        layers = []
        if fc_layers_config:
            current_dim = input_dim
            for i, layer_config in enumerate(fc_layers_config):
                layer_output_dim = layer_config.get('output_size', output_dim if i == len(fc_layers_config) - 1 else current_dim)
                layers.append(nn.Linear(current_dim, layer_output_dim, bias=use_bias))
                if i < len(fc_layers_config) - 1:
                    if activation == "relu":
                        layers.append(nn.ReLU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
                    elif activation == "sigmoid":
                        layers.append(nn.Sigmoid())
                    elif activation == "leaky_relu":
                        layers.append(nn.LeakyReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if norm == "batch":
                    layers.append(nn.BatchNorm1d(layer_output_dim))
                elif norm == "layer":
                    layers.append(nn.LayerNorm(layer_output_dim))
                current_dim = layer_output_dim
        else:
            if num_layers == 1:
                layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            else:
                intermediate_dims = [input_dim]
                for i in range(num_layers - 1):
                    intermediate_dim = int(input_dim * (0.5 ** (i + 1)))
                    intermediate_dim = max(intermediate_dim, output_dim)
                    intermediate_dims.append(intermediate_dim)
                intermediate_dims.append(output_dim)
                for i in range(num_layers):
                    layers.append(nn.Linear(intermediate_dims[i], intermediate_dims[i + 1], bias=use_bias))
                    if i < num_layers - 1:
                        if activation == "relu":
                            layers.append(nn.ReLU())
                        elif activation == "tanh":
                            layers.append(nn.Tanh())
                        elif activation == "sigmoid":
                            layers.append(nn.Sigmoid())
                        elif activation == "leaky_relu":
                            layers.append(nn.LeakyReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    if norm == "batch":
                        layers.append(nn.BatchNorm1d(intermediate_dims[i + 1]))
                    elif norm == "layer":
                        layers.append(nn.LayerNorm(intermediate_dims[i + 1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] != self.expected_channels:
            if (
                self.channel_adapter is None
                or self.channel_adapter.in_channels != x.shape[1]
                or self.channel_adapter.out_channels != self.expected_channels
            ):
                self.channel_adapter = nn.Conv2d(
                    x.shape[1],
                    self.expected_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ).to(x.device)
                logger.info(
                    "Created dynamic channel adapter: %s -> %s channels",
                    x.shape[1],
                    self.expected_channels,
                )
            x = self.channel_adapter(x)

        target_height, target_width = self.height, self.width
        if x.shape[2] != target_height or x.shape[3] != target_width:
            if (
                self.size_adapter is None
                or getattr(self.size_adapter, "output_size", None)
                != (target_height, target_width)
            ):
                self.size_adapter = nn.AdaptiveAvgPool2d(
                    (target_height, target_width)
                ).to(x.device)
                logger.info(
                    "Created size adapter: %sx%s -> %sx%s",
                    x.shape[2],
                    x.shape[3],
                    target_height,
                    target_width,
                )
            x = self.size_adapter(x)

        if target_height != self.expected_height or target_width != self.expected_width:
            if (
                self.backbone_adapter is None
                or getattr(self.backbone_adapter, "output_size", None)
                != (self.expected_height, self.expected_width)
            ):
                self.backbone_adapter = nn.AdaptiveAvgPool2d(
                    (self.expected_height, self.expected_width)
                ).to(x.device)
                logger.info(
                    "Aligning to MetaFormer backbone size: %sx%s",
                    self.expected_height,
                    self.expected_width,
                )
            x = self.backbone_adapter(x)

        features = self.backbone.forward_features(x)
        output = self.fc_layers(features)
        return {'encoder_output': output}

    @property
    def output_shape(self):
        return [self.output_size]


def create_metaformer_stacked_cnn(model_name: str, **kwargs) -> MetaFormerStackedCNN:
    encoder = MetaFormerStackedCNN(custom_model=model_name, **kwargs)
    return encoder


def patch_ludwig_stacked_cnn():
    # Only patch Ludwig if MetaFormer models are available in this runtime
    if not META_MODELS_AVAILABLE:
        logger.warning("MetaFormer models unavailable; skipping Ludwig patch for stacked_cnn.")
        return False
    return patch_ludwig_direct()


def _is_supported_metaformer(custom_model: Optional[str]) -> bool:
    return bool(custom_model) and custom_model.startswith(SUPPORTED_PREFIXES)


def patch_ludwig_direct():
    try:
        from ludwig.encoders.image.base import Stacked2DCNN
        original_stacked_cnn_init = Stacked2DCNN.__init__

        def patched_stacked_cnn_init(self, *args, **kwargs):
            custom_model = kwargs.pop("custom_model", None)
            if custom_model is None:
                custom_model = getattr(patch_ludwig_direct, '_metaformer_model', None)

            try:
                if META_MODELS_AVAILABLE and _is_supported_metaformer(custom_model):
                    print(f"DETECTED MetaFormer model: {custom_model}")
                    print("MetaFormer encoder is being loaded and used.")
                    # Initialize base class to keep Ludwig internals intact
                    original_stacked_cnn_init(self, *args, **kwargs)
                    # Create our MetaFormer encoder and graft behavior
                    mf_encoder = create_metaformer_stacked_cnn(custom_model, **kwargs)
                    # ensure base attributes won't be used accidentally
                    for attr in ("conv_layers", "fc_layers", "combiner", "output_shape", "reduce_output"):
                        if hasattr(self, attr):
                            try:
                                setattr(self, attr, getattr(mf_encoder, attr, None))
                            except Exception:
                                pass
                    self.forward = mf_encoder.forward
                    if hasattr(mf_encoder, 'backbone'):
                        self.backbone = mf_encoder.backbone
                    if hasattr(mf_encoder, 'fc_layers'):
                        self.fc_layers = mf_encoder.fc_layers
                    if hasattr(mf_encoder, 'custom_model'):
                        self.custom_model = mf_encoder.custom_model
                    # explicit confirmation logs
                    try:
                        url_info = getattr(mf_encoder, '_loaded_weights_url', None)
                        loaded_flag = getattr(mf_encoder, '_pretrained_loaded', False)
                        if loaded_flag and url_info:
                            print(f"CONFIRMED: MetaFormer '{custom_model}' using pretrained weights from: {url_info}")
                            logger.info(f"CONFIRMED: MetaFormer '{custom_model}' using pretrained weights from: {url_info}")
                        else:
                            print(f"CONFIRMED: MetaFormer '{custom_model}' using randomly initialized weights (no pretrained)")
                            logger.info(f"CONFIRMED: MetaFormer '{custom_model}' using randomly initialized weights")
                    except Exception:
                        pass
                else:
                    original_stacked_cnn_init(self, *args, **kwargs)
            finally:
                if hasattr(patch_ludwig_direct, '_metaformer_model'):
                    patch_ludwig_direct._metaformer_model = None

        Stacked2DCNN.__init__ = patched_stacked_cnn_init
        return True
    except Exception as e:
        logger.error(f"Failed to apply MetaFormer direct patch: {e}")
        return False


def set_current_metaformer_model(model_name: str):
    """Store the current MetaFormer model name for the patch to use."""
    setattr(patch_ludwig_direct, '_metaformer_model', model_name)


def clear_current_metaformer_model():
    """Remove any cached MetaFormer model hint."""
    if hasattr(patch_ludwig_direct, '_metaformer_model'):
        delattr(patch_ludwig_direct, '_metaformer_model')
