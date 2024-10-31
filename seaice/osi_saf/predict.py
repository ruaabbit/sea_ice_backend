from typing import List

import numpy as np
import torch
from PIL import Image

from .config import configs
from .utils.model_factory import IceNet

model_path: str = "seaice/osi_saf/checkpoints/checkpoint_SICFN_14.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IceNet().to(device)
checkpoint = torch.load(model_path, weights_only=True, map_location=device)
model.load_state_dict(checkpoint["net"])
model.eval()


def predict_ice_concentration(
        image_list: List[Image.Image],
) -> np.ndarray:
    """
    Predict future sea ice concentration from input images.
    Args:
        image_list: List of 14 PIL Image objects containing sea ice concentration maps
    Returns:
        np.ndarray: Predicted sea ice concentration maps for next 14 days
                   Shape: (14, H, W) where H and W are the original image dimensions
    """

    # Process images
    processed_images = []
    for img in image_list:
        # Convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')
        # Resize to model input size
        img = img.resize(configs.img_size, Image.Resampling.BILINEAR)
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        processed_images.append(img_array)

    # Stack images and add batch and channel dimensions
    input_tensor = torch.from_numpy(np.stack(processed_images))[None, :, None, :, :]
    input_tensor = input_tensor.float().to(device)

    # Create dummy targets (will be ignored during inference)
    dummy_targets = torch.zeros_like(input_tensor)

    # Generate predictions
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            predictions, _ = model(input_tensor, dummy_targets)

    # Convert predictions to numpy array
    predictions = predictions.detach().to(torch.float).cpu().numpy()[0]
    print(predictions.shape)

    return predictions
