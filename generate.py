import sys

import matplotlib.pyplot as plt
import noise
import numpy as np
import torch
import torchvision
from PIL import Image
from omegaconf import OmegaConf

from gradient_noise import perline_noise, show_image

sys.path.append('taming-transformers')

from taming.models import cond_transformer, vqgan
from CLIP import clip

device = torch.device("cpu")


# clip model
def load_clip_model():
    jit = True if "1.7.1" in torch.__version__ else False
    model = clip.load('ViT-B/32', jit=jit)[0].eval()
    model.requires_grad_(False).to(device)

    print(clip.available_models())
    print('clip model visual input resolution: ', model.visual.input_resolution)
    return model


config_path = "taming-transformers/checkpoints/vqgan_imagenet_f16_16384.yaml"
checkpoint_path = "taming-transformers/checkpoints/vqgan_imagenet_f16_16384.ckpt"


def load_vqgan_model():
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model.to(device)


vqgan_model = load_vqgan_model()
clip_model = load_clip_model()
normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])
batch_size = 1
height = 100
width = 100
shape = (width, height, 3)

init_image = perline_noise(shape)
show_image(init_image)


def encode_text(text):
    t = clip.tokenize(text).cuda()
    t = clip_model.encode_text(t).detach().clone()
    return t


def create_encodings(text_list):
    encodings = []
    for text in text_list:
        encodings.append(encode_text(text))
    return encodings


aug_trans = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ColorJitter(hue=0.01, saturation=0.01, brightness=0.01, contrast=0.01),
    torchvision.transforms.RandomAdjustSharpness(0.3, p=0.4),
    torchvision.transforms.RandomAffine(30, (0.1, 0.1)),
    torchvision.transforms.RandomPerspective(0.2, p=0.4), ).cuda()

image = aug_trans(init_image)
# dis = Image.fromarray(image)
show_image(image)
