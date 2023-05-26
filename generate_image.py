import sys

from matplotlib import cm
from tqdm import tqdm

sys.path.append('taming-transformers')

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import torchvision.transforms as T

from CLIP import clip
import numpy as np
import noise
import matplotlib.pyplot as plt
from PIL import Image


class GenerateImage:

    def __init__(self, nb=False, init_image=None, prompts=[]):
        self.cut_size = None
        self.out_tensor = None
        self.nb = nb
        self.z_max = None
        self.z_min = None
        self.optimizer = None
        self.encoded_prompts = []
        self.clip_model = None
        self.z_orig = None
        self.vqgan_model = None
        self.z = None
        self.output_image_size = [256, 256]
        self.init_image = init_image
        self.init_noise = None
        self.iterations = 180
        self.init_image_weight = 0.0
        self.clip_model_name = 'ViT-B/32'
        self.learning_rate = 0.08 if self.init_image is None else 0.05
        self.start_learning_rate = 0.2 if self.init_image is None else 0.1
        self.num_cuts = 64
        self.prompts = prompts
        # if self.nb:
        #     self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # else:
        #     self.device = torch.device('cpu')
        self.device = torch.device('cuda:0')

        self.replace_grad = ReplaceGrad.apply
        self.clamp_with_grad = ClampWithGrad.apply

        self.seed = torch.seed()
        torch.manual_seed(self.seed)

        self.augs = nn.Sequential(
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAdjustSharpness(0.3, p=0.4),
            self._TRandom(T.RandomAffine(degrees=30, translate=(0.1, 0.1)), 0.8),
            self._TRandom(T.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01), p=0.7),
            T.RandomPerspective(distortion_scale=0.2, p=0.4),
            T.RandomGrayscale(p=0.07),
            # T.RandomErasing(scale=(0.1, 0.33), ratio=(0.3, 3.3), p=0.4),
        )

    def initialize(self):
        self.load_vqgan_model()
        self.load_clip_model()

    def generate(self, prompts, img=None, style="default"):
        # self.prompts = prompts
        if img is not None:
            self.iterations = 50
        self.init_image = img
        self.init_z()
        self.show_init_z()
        self.optimizer = optim.AdamW([self.z], lr=self.learning_rate)
        print(style)
        if style == 'Epic':
            prompts.append((
                           ' Epic cinematic brilliant stunning intricate meticulously detailed dramatic atmospheric maximalist digital matte painting',
                           0.9))
        elif style == 'Dark Fantasy':
            prompts.append((
                           '  a masterpiece, 8k resolution, dark fantasy concept art, by Greg Rutkowski, dynamic lighting, hyperdetailed, intricately detailed, Splash screen art, trending on Artstation, deep color, Unreal Engine, volumetric lighting, Alphonse Mucha, Jordan Grimmer, purple and yellow complementary colours',
                           0.9))
        elif style == 'Sinister':
            prompts.append(('  sinister by Greg Rutkowski', 0.9))
        elif style == 'Fantasy':
            prompts.append(('  ethereal fantasy hyperdetailed mist Thomas Kinkade', 0.9))
        elif style == 'Horror':
            prompts.append(('  horror Gustave Doré Greg Rutkowski', 0.9))
        elif style == 'Surreal':
            prompts.append(('  surrealism Salvador Dali matte background melting oil on canvas', 0.9))
        elif style == 'Steampunk':
            prompts.append(('  steampunk engine', 0.9))
        elif style == 'Cyberpunk':
            prompts.append((' cyberpunk 2099 blade runner 2049 neon', 0.9))
        elif style == 'Synthwave':
            prompts.append(('synthwave neon retro', 0.9))
        elif style == 'Heavenly':
            prompts.append(('  heavenly sunshine beams divine bright soft focus holy in the clouds', 0.9))
        else:
            prompts.append(('high resolution', 0.9))

        self.encode_text(prompts)

        try:
            for i in tqdm(range(1, self.iterations + 1), unit='iteration', leave=True):
                loss_all = self.optimize()
                if (i <= 20 and i % 5 == 0) or i % 20 == 0:
                    str = ', '.join(f'{loss.item():7.3f}' for loss in loss_all)
                    tqdm.write(f'i:{i:03}\tloss sum: {sum(loss_all).item():7.3f}\tloss for each prompt:{str}')
                    img = self.get_current_image()
                    # if self.nb:
                    #     display(img)
                    # else:
                    #     img.save(f"img{i:03}.png")
            plt.imshow(self.get_current_image())
            plt.show()
        except KeyboardInterrupt:
            pass
        return img

    def load_vqgan_model(self):
        config_path = "checkpoints/vqgan_imagenet_f16_16384.yaml"
        checkpoint_path = "checkpoints/vqgan_imagenet_f16_16384.ckpt"

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
        self.vqgan_model = model.to(self.device)
        self.z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    def load_clip_model(self):
        jit = True if "1.7.1" in torch.__version__ else False
        model = clip.load('ViT-B/32', jit=jit)[0].eval()
        model.requires_grad_(False).to(self.device)

        print(clip.available_models())
        print('clip model visual input resolution: ', model.visual.input_resolution)
        self.clip_model = model
        self.cut_size = self.clip_model.visual.input_resolution  # 224

    def init_z(self):
        if self.init_image is not None:
            self.z_from_image(self.init_image)
        else:
            self.z_from_image(self.perlin_noise())
        # self.z_from_image(self.init_image)

    def perlin_noise(self):
        scale = 100
        octaves = 10
        persistence = 0.618
        lacunarity = 1.618
        shape = self.output_image_size
        a = np.zeros(shape)
        seed = np.random.randint(0, 100)
        r = np.random.randint(100, 200)
        print(seed, r)
        for i in range(shape[0]):
            for j in range(shape[1]):
                a[i][j] = noise.pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=r,
                                        repeaty=r,
                                        base=seed)

        a = (a - np.min(a)) / np.ptp(a)
        img = Image.fromarray(np.uint8(cm.gist_earth(a) * 255))
        return img

    def z_random(self):
        x, y = self.vqgan_out_size()
        e_dim = self.vqgan_model.quantize.e_dim
        n_toks = self.vqgan_model.quantize.n_e
        one_hot = F.one_hot(torch.randint(n_toks, [y * x], device=self.device), n_toks).float()
        self.z = one_hot @ self.vqgan_model.quantize.embedding.weight
        self.z = self.z.view([-1, y, x, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)

    def vqgan_out_size(self):
        f = 2 ** (self.vqgan_model.decoder.num_resolutions - 1)
        x = (self.output_image_size[0] // f) * f
        y = (self.output_image_size[1] // f) * f
        return x, y

    def z_from_image(self, image):
        x, y = self.vqgan_out_size()
        if self.init_image is not None:
            image = Image.fromarray(image)
        image = image.convert('RGB')
        image = image.resize((x, y), Image.LANCZOS)
        pil_tensor = TF.to_tensor(image)
        self.z, *_ = self.vqgan_model.encode(pil_tensor.to(self.device).unsqueeze(0) * 2 - 1)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)

    def prompt_loss(self, encoded_img, encoded_txt, weight=torch.as_tensor(1.0)):
        encoded_img = F.normalize(encoded_img.unsqueeze(1), dim=2)
        encoded_txt = F.normalize(encoded_txt.unsqueeze(0), dim=2)
        d = encoded_img.sub(encoded_txt).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        d = d * weight.sign()
        return weight.abs() * self.replace_grad(d, d).mean()

    def vector_quantize(self, x, codebook):
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return self.replace_grad(x_q, x)

    def synth(self, z):
        z_q = self.vector_quantize(z.movedim(1, 3), self.vqgan_model.quantize.embedding.weight).movedim(3, 1)
        return self.clamp_with_grad(self.vqgan_model.decode(z_q).add(1).div(2), 0, 1)

    normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])

    def total_loss(self):
        self.out_tensor = self.synth(self.z)
        out = self.make_crops(self.out_tensor)
        out = self.normalize(out)
        encoded_img = self.clip_model.encode_image(out).float()
        loss = []
        for p in self.encoded_prompts:
            w = torch.as_tensor(p[1])
            loss.append(self.prompt_loss(encoded_img, p[0], w))
        return loss

    def show_init_z(self):
        out_tensor = self.synth(self.z)
        img = TF.to_pil_image(out_tensor[0].cpu())
        if self.nb:
            display(img)
        else:
            img.save(f"img000.png")

    def get_current_image(self):
        return TF.to_pil_image(self.out_tensor[0].cpu())

    def optimize(self):
        self.optimizer.zero_grad(set_to_none=True)
        lossAll = self.total_loss()
        loss = sum(lossAll)
        loss.backward()
        self.optimizer.step()

        # with torch.no_grad():
        with torch.inference_mode():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

        return lossAll

    def encode_text(self, prompts):
        encoded = []
        for p in prompts:
            txt = clip.tokenize(p[0]).to(self.device)
            encoded_txt = self.clip_model.encode_text(txt).float()
            encoded.append((encoded_txt, p[1]))
        self.encoded_prompts = encoded

    def make_crops(self, input):
        sideY, sideX = input.shape[2:4]
        noise_fac = 0.1
        cutouts = []
        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size / min_size_width)

        for _ in range(self.num_cuts):
            rand_size = torch.zeros(1, ).normal_(mean=.8, std=.3).clip(lower_bound, 1.)
            size_mult = rand_size ** 1
            size = int(min_size_width * (size_mult.clip(lower_bound, 1.)))
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.interpolate(cutout, (self.cut_size, self.cut_size), mode='bicubic', align_corners=True)
            cutouts.append(cutout)

        cutouts = torch.cat(cutouts, dim=0)
        cutouts = self.clamp_with_grad(cutouts, 0, 1)
        cutouts = self.augs(cutouts)
        facs = cutouts.new_empty([self.num_cuts, 1, 1, 1]).uniform_(0, noise_fac)
        cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts

    @staticmethod
    def _TRandom(transform, p):
        return T.RandomApply(nn.ModuleList([transform, ]), p=p)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
