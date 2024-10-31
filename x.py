import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


def upscale(image_path, imgtarget):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealESRGAN(device, scale=2)
    model.load_weights("weights/RealESRGAN_x2.pth", download=True)
    image = Image.open(image_path).convert("RGB")
    sr_image = model.predict(image)
    sr_image.save(imgtarget)


upscale("img.jpg", "img2.jpg")
