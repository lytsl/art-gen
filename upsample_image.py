import sys

sys.path.append('Real-ESRGAN')

from realesrgan import RealESRGAN
from PIL import Image
import numpy as np
import torch


def upscale(img, device=None):
    if device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=4)
    model.load_weights('/content/RealESRGAN/weights/RealESRGAN_x4.pth')

    return model.predict(np.array(img))



