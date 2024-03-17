from diffusers import StableDiffusionXLPipeline
import os
import random
import torch


def initialize_txt2img(model_name, model_path, device, torch_dtype):
    "initialize the txt2img model"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)

    if not (os.path.isdir(model_path)):
        pipe.save_pretrained(model_path)

    return pipe


def gen_txt2img(
    pipe,
    prompt,
    device,
    image_name="output",
    num_variations=1,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512,
    manual_seeds=None,
    negative_prompt="",
):
    "generate an image"
    if manual_seeds is None:
        manual_seeds = [random.randint(0, 65000) for _ in range(num_variations)]
    generator = [torch.Generator(device).manual_seed(i) for i in manual_seeds]

    images = pipe(
        prompt=num_variations * [prompt],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        negative_prompt=num_variations * [negative_prompt],
    ).images

    img_paths = []
    for i in range(len(images)):
        img_path = f"metadata/output_images/{image_name}_seed_{manual_seeds[i]}.png"
        images[i].save(img_path)
        img_paths.append(img_path)

    return img_paths
