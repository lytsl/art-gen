## Art Gen

This project utilizes the VQGAN+CLIP image generation model, complemented by ESRGAN's upscaling capabilities, to create high-quality images. The VQGAN+CLIP image generation model is a powerful combination of two separate models that work together to create stunning images. VQGAN (Vector Quantized Generative Adversarial Network) is responsible for generating high-quality images by mapping a low-dimensional noise vector to an image space through the use of vector quantization. On the other hand, CLIP (Contrastive Language Image Pre-Training) provides a way to match the generated images with the original prompts or text descriptions used as input.

Together, VQGAN and CLIP form a symbiotic relationship where the strengths of one complement the other, resulting in highly detailed and accurate images. The addition of ESRGAN for upscaling further enhances the quality of the final output, ensuring that the small details are preserved even after increasing the resolution.

## Installation

To install the necessary dependencies, follow the below steps:

1. Install the `environment.yml` file with Anaconda.

2. Clone the `Real-ESRGAN`,`CLIP` and `taming-transformers` git repositories by running the following commands:
    ```
    git clone https://github.com/sberbank-ai/Real-ESRGAN
    git clone https://github.com/openai/CLIP.git
    git clone https://github.com/CompVis/taming-transformers.git
    ```
    
3. Download the `VQGAN` model by going into the `taming-transformers` directory and running the following commands:
    ```
    mkdir -p checkpoints
    wget 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' -O 'checkpoints/vqgan_imagenet_f16_16384.ckpt'
    mkdir checkpoints
    wget 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' -O 'checkpoints/vqgan_imagenet_f16_16384.yaml'
    ``` 
    
4. Download the `RealESRGAN` model by going into the `RealESRGAN` directory and running the following commands:
    ```
    gdown https://drive.google.com/uc?id=1SGHdZAln4en65_NQeQY9UjchtkEF9f5F -O weights/RealESRGAN_x4.pth &> /dev/null
    ``` 
    
5. Run the `generate_image.py` file or the `gen.ipynb` notebook to generate images based on your input.


## Result

|Text Prompt |Style |Output |
|--- |---|---|
|a castle above clouds  |heavenly |![image](https://github.com/lytsl/art-gen/assets/85685866/c44f47d7-df85-4283-9711-60bb3e74b193) |
|scream |horror |![image](https://github.com/lytsl/art-gen/assets/85685866/5cb09460-d620-4aa0-9743-f1d1f4a8d53a) |
|Beautiful desolated majestic futuristic technologically advanced cyberpunk scientist civilization |dark fantasy |![image](https://github.com/lytsl/art-gen/assets/85685866/13c2dcbb-034f-4950-99ab-219aa79b99f2)|
|keanu reeves |default |![image](https://github.com/lytsl/art-gen/assets/85685866/220ed1e0-ffcb-4cb6-9f32-19d8b55e91b0)|
|a painting of sunset at sea |epic |![image](https://github.com/lytsl/art-gen/assets/85685866/1d38eb59-66e7-483b-b1e3-26dc44406628) |


## Credits

This project was inspired by Katherine Crowson's original implementation of the VQGAN+CLIP model.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
