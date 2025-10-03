"""
This module provides functionality to extract image embeddings
using a specified
pretrained model from the torchvision library. It includes functions to:
- List image files directly from a ZIP file without extraction.
- Apply model-specific preprocessing and transformations.
- Extract embeddings using various models.
- Save the resulting embeddings into a CSV file.
Modules required:
- argparse: For command-line argument parsing.
- os, csv, zipfile: For file handling (ZIP file reading, CSV writing).
- inspect: For inspecting function signatures and models.
- torch, torchvision: For loading and using pretrained models
to extract embeddings.
- PIL, cv2: For image processing tasks such as resizing, normalization,
and conversion.
"""

import argparse
import csv
import inspect
import logging
import os
import zipfile
from inspect import signature

import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# GPFM imports
try:
    import requests
    GPFM_AVAILABLE = True
except ImportError as e:
    GPFM_AVAILABLE = False
    logging.warning(f"GPFM dependencies not available: {e}")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("/tmp/ludwig_embeddings.log", mode="a")  # File output
    ]
)

# Create a cache directory in the current working directory
cache_dir = os.path.join(os.getcwd(), 'hf_cache')
try:
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"Cache directory created: {cache_dir}, writable: {os.access(cache_dir, os.W_OK)}")
except OSError as e:
    logging.error(f"Failed to create cache directory {cache_dir}: {e}")
    raise

# GPFM DinoVisionTransformer Implementation


class DinoVisionTransformer(torch.nn.Module):
    """Simplified DinoVisionTransformer for GPFM."""

    def __init__(self, img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = torch.nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        # Class token
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks (simplified)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.0,
                batch_first=True
            ) for _ in range(depth)
        ])

        # Layer norm
        self.norm = torch.nn.LayerNorm(embed_dim)

        # Initialize weights
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply layer norm and return class token
        x = self.norm(x)
        return x[:, 0]  # Return class token features


# GPFM Model Implementation
class GPFMModel(torch.nn.Module):
    """GPFM (Generalizable Pathology Foundation Model) implementation."""

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.model = None
        self.transformer = None
        self.embed_dim = 1024  # GPFM uses 1024-dimensional embeddings
        self._load_model()

    def _download_weights(self, url, filepath):
        """Download GPFM weights from the official repository."""
        if os.path.exists(filepath):
            logging.info(f"GPFM weights already exist at {filepath}")
            return True

        logging.info(f"Downloading GPFM weights from {url}")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                                logging.info(f"Downloaded {downloaded // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB ({progress:.1f}%)")

            logging.info(f"GPFM weights downloaded successfully to {filepath}")
            return True

        except Exception as e:
            logging.error(f"Failed to download GPFM weights: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up partial download
            return False

    def _load_model(self):
        """Load GPFM model with pretrained weights."""
        try:
            # Create models directory
            models_dir = os.path.join(cache_dir, 'gpfm_models')
            os.makedirs(models_dir, exist_ok=True)

            # GPFM weights URL from official repository
            weights_url = "https://github.com/birkhoffkiki/GPFM/releases/download/ckpt/GPFM.pth"
            weights_path = os.path.join(models_dir, 'GPFM.pth')

            # Create GPFM DinoVisionTransformer architecture
            self.model = DinoVisionTransformer(
                img_size=224,
                patch_size=14,
                embed_dim=1024,
                depth=24,
                num_heads=16
            )

            # Try to download and load GPFM weights
            weights_loaded = False
            if self._download_weights(weights_url, weights_path):
                try:
                    logging.info("Loading GPFM pretrained weights...")
                    checkpoint = torch.load(weights_path, map_location=self.device)

                    # Extract teacher model weights (GPFM format)
                    if 'teacher' in checkpoint:
                        state_dict = checkpoint['teacher']
                        logging.info("Found 'teacher' key in checkpoint")
                    else:
                        state_dict = checkpoint
                        logging.info("Using checkpoint directly")

                    # Rename keys to match our simplified architecture
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        # Remove 'backbone.' prefix if present
                        if k.startswith('backbone.'):
                            k = k[9:]  # Remove 'backbone.'

                        # Map GPFM keys to our simplified architecture
                        if k in ['cls_token', 'pos_embed']:
                            new_state_dict[k] = v
                        elif k.startswith('patch_embed.proj.'):
                            # Map patch embedding
                            new_k = k.replace('patch_embed.proj.', 'patch_embed.')
                            new_state_dict[new_k] = v
                        elif k.startswith('blocks.') and 'norm' in k:
                            # Map layer norms
                            if k.endswith('.norm1.weight') or k.endswith('.norm1.bias'):
                                # Skip intermediate norms for simplified model
                                continue
                            elif k.endswith('.norm2.weight') or k.endswith('.norm2.bias'):
                                continue
                        elif k == 'norm.weight' or k == 'norm.bias':
                            new_state_dict[k] = v

                    # Load compatible weights
                    missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                    if missing_keys:
                        logging.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                    if unexpected_keys:
                        logging.warning(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5

                    logging.info("GPFM pretrained weights loaded successfully")
                    weights_loaded = True

                except Exception as e:
                    logging.warning(f"Could not load GPFM weights: {e}")

            if not weights_loaded:
                logging.info("Using randomly initialized GPFM architecture (no pretrained weights)")

            self.model = self.model.to(self.device)
            self.model.eval()

            # GPFM preprocessing (based on official repository)
            self.transformer = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),  # Ensure RGB format
                transforms.Resize((224, 224)),  # GPFM uses 224x224 (not 512x512 for features)
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225]
                )
            ])

            logging.info(f"GPFM model initialized successfully (embed_dim: {self.embed_dim})")

        except Exception as e:
            logging.error(f"Failed to initialize GPFM model: {e}")
            raise

    def forward(self, x):
        """Forward pass through GPFM model."""
        with torch.no_grad():
            return self.model(x)

    def get_transformer(self):
        """Get the preprocessing transformer for GPFM."""
        return self.transformer


# Available models from torchvision
AVAILABLE_MODELS = {
    name: getattr(models, name)
    for name in dir(models)
    if callable(
        getattr(models, name)
    ) and "weights" in signature(getattr(models, name)).parameters
}

# Add GPFM model if available
if GPFM_AVAILABLE:
    AVAILABLE_MODELS['gpfm'] = GPFMModel

# Default resize and normalization settings for models
MODEL_DEFAULTS = {
    "default": {"resize": (224, 224), "normalize": (
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )},
    "efficientnet_b1": {"resize": (240, 240)},
    "efficientnet_b2": {"resize": (260, 260)},
    "efficientnet_b3": {"resize": (300, 300)},
    "efficientnet_b4": {"resize": (380, 380)},
    "efficientnet_b5": {"resize": (456, 456)},
    "efficientnet_b6": {"resize": (528, 528)},
    "efficientnet_b7": {"resize": (600, 600)},
    "inception_v3": {"resize": (299, 299)},
    "swin_b": {"resize": (224, 224), "normalize": (
        [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]
    )},
    "swin_s": {"resize": (224, 224), "normalize": (
        [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]
    )},
    "swin_t": {"resize": (224, 224), "normalize": (
        [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]
    )},
    "vit_b_16": {"resize": (224, 224), "normalize": (
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    )},
    "vit_b_32": {"resize": (224, 224), "normalize": (
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    )},
    "gpfm": {
        "resize": (224, 224),
        "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    },
}

for model, settings in MODEL_DEFAULTS.items():
    if "normalize" not in settings:
        settings["normalize"] = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# Custom transform classes
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img):
        img = np.array(img.convert("L"))
        img = self.clahe.apply(img)
        return Image.fromarray(img).convert("RGB")


class CannyTransform:
    def __init__(self, threshold1=100, threshold2=200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def __call__(self, img):
        img = np.array(img.convert("L"))
        edges = cv2.Canny(img, self.threshold1, self.threshold2)
        return Image.fromarray(edges).convert("RGB")


class RGBAtoRGBTransform:
    def __call__(self, img):
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img).convert("RGB")
        else:
            img = img.convert("RGB")
        return img


def get_image_files_from_zip(zip_file):
    """Returns a list of image file names in the ZIP file."""
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            file_list = [
                f for f in zip_ref.namelist() if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif")
                )
            ]
        return file_list
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Invalid ZIP file.") from exc
    except Exception as exc:
        raise RuntimeError("Error reading ZIP file.") from exc


def load_model(model_name, device):
    """Loads a specified torchvision model and
    modifies it for feature extraction."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. \
            Available models: {list(AVAILABLE_MODELS.keys())}")
    try:
        # Special handling for GPFM
        if model_name == "gpfm":
            model = AVAILABLE_MODELS[model_name](device=device)
            logging.info("GPFM model loaded")
            return model

        # Standard torchvision models
        if "weights" in inspect.signature(
                AVAILABLE_MODELS[model_name]).parameters:
            model = AVAILABLE_MODELS[model_name](weights="DEFAULT").to(device)
        else:
            model = AVAILABLE_MODELS[model_name]().to(device)
        logging.info("Model loaded")
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        raise

    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()
    elif hasattr(model, "head"):
        model.head = torch.nn.Identity()

    model.eval()
    return model


def write_csv(output_csv, list_embeddings, ludwig_format=False):
    """Writes embeddings to a CSV file, optionally in Ludwig format."""
    with open(output_csv, mode="w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if list_embeddings:
            if ludwig_format:
                header = ["sample_name", "embedding"]
                formatted_embeddings = []
                for embedding in list_embeddings:
                    sample_name = embedding[0]
                    vector = embedding[1:]
                    embedding_str = " ".join(map(str, vector))
                    formatted_embeddings.append([sample_name, embedding_str])
                csv_writer.writerow(header)
                csv_writer.writerows(formatted_embeddings)
                logging.info("CSV created in Ludwig format")
            else:
                header = ["sample_name"] + [f"vector{i + 1}" for i in range(
                    len(list_embeddings[0]) - 1
                )]
                csv_writer.writerow(header)
                csv_writer.writerows(list_embeddings)
                logging.info("CSV created")
        else:
            csv_writer.writerow(["sample_name"] if not ludwig_format
                                else ["sample_name", "embedding"])
            logging.info("No valid images found. Empty CSV created.")


def extract_embeddings(
        model_name,
        apply_normalization,
        zip_file,
        file_list,
        transform_type="rgb"):
    """Extracts embeddings from images
    using batch processing or sequential fallback."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, device)
    model_settings = MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["default"])
    resize = model_settings["resize"]
    normalize = model_settings.get("normalize", (
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ))

    # Define transform pipeline
    if transform_type == "grayscale":
        initial_transform = transforms.Grayscale(num_output_channels=3)
    elif transform_type == "clahe":
        initial_transform = CLAHETransform()
    elif transform_type == "edges":
        initial_transform = CannyTransform()
    elif transform_type == "rgba_to_rgb":
        initial_transform = RGBAtoRGBTransform()
    else:
        initial_transform = transforms.Lambda(lambda x: x.convert("RGB"))

    # Handle GPFM separately as it has its own preprocessing
    if model_name == "gpfm":
        # For GPFM, combine initial transform with GPFM's custom transformer
        if transform_type in ["grayscale", "clahe", "edges", "rgba_to_rgb"]:
            transform = transforms.Compose([
                initial_transform,
                model.get_transformer()
            ])
        else:
            transform = model.get_transformer()
    else:
        # Standard torchvision models
        transform_list = [initial_transform,
                          transforms.Resize(resize),
                          transforms.ToTensor()]
        if apply_normalization:
            transform_list.append(transforms.Normalize(mean=normalize[0],
                                                       std=normalize[1]))
        transform = transforms.Compose(transform_list)

    class ImageDataset(Dataset):
        def __init__(self, zip_file, file_list, transform=None):
            self.zip_file = zip_file
            self.file_list = file_list
            self.transform = transform

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            with zipfile.ZipFile(self.zip_file, "r") as zip_ref:
                with zip_ref.open(self.file_list[idx]) as file:
                    try:
                        image = Image.open(file)
                        if self.transform:
                            image = self.transform(image)
                        return image, os.path.basename(self.file_list[idx])
                    except Exception as e:
                        logging.warning(
                            "Skipping %s: %s", self.file_list[idx], e
                        )
                        return None, os.path.basename(self.file_list[idx])

    # Custom collate function
    def collate_fn(batch):
        batch = [item for item in batch if item[0] is not None]
        if not batch:
            return None, None
        images, names = zip(*batch)
        return torch.stack(images), names

    list_embeddings = []
    with torch.inference_mode():
        try:
            # Try DataLoader with reduced resource usage
            dataset = ImageDataset(zip_file, file_list, transform=transform)
            dataloader = DataLoader(
                dataset,
                batch_size=16,  # Reduced for lower memory usage
                num_workers=0,  # Fix multiprocessing issues with GPFM
                shuffle=False,
                pin_memory=True if device == "cuda" else False,
                collate_fn=collate_fn,
            )
            for images, names in dataloader:
                if images is None:
                    continue
                images = images.to(device)
                embeddings = model(images).cpu().numpy()
                for name, embedding in zip(names, embeddings):
                    list_embeddings.append([name] + embedding.tolist())
        except RuntimeError as e:
            logging.warning(
                f"DataLoader failed: {e}. \
                Falling back to sequential processing."
            )
            # Fallback to sequential processing
            for file in file_list:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    with zip_ref.open(file) as img_file:
                        try:
                            image = Image.open(img_file)
                            image = transform(image)
                            input_tensor = image.unsqueeze(0).to(device)
                            embedding = model(
                                input_tensor
                            ).squeeze().cpu().numpy()
                            list_embeddings.append(
                                [os.path.basename(file)] + embedding.tolist()
                            )
                        except Exception as e:
                            logging.warning("Skipping %s: %s", file, e)

    return list_embeddings


def main(zip_file, output_csv, model_name, apply_normalization=False,
         transform_type="rgb", ludwig_format=False):
    """Main entry point for processing the zip file and
    extracting embeddings."""
    file_list = get_image_files_from_zip(zip_file)
    logging.info("Image files listed from ZIP")

    list_embeddings = extract_embeddings(
        model_name,
        apply_normalization,
        zip_file,
        file_list,
        transform_type
    )
    logging.info("Embeddings extracted")
    write_csv(output_csv, list_embeddings, ludwig_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image embeddings.")
    parser.add_argument(
        "--zip_file",
        required=True,
        help="Path to the ZIP file containing images."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        choices=AVAILABLE_MODELS.keys(),
        help="Model for embedding extraction."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to apply normalization."
    )
    parser.add_argument(
        "--transform_type",
        required=True,
        help="Image transformation type."
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to the output CSV file"
    )
    parser.add_argument(
        "--ludwig_format",
        action="store_true",
        help="Prepare CSV file in Ludwig input format"
    )

    args = parser.parse_args()
    main(
        args.zip_file,
        args.output_csv,
        args.model_name,
        args.normalize,
        args.transform_type,
        args.ludwig_format
    )
