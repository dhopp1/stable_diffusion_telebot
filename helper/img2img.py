from diffusers import StableDiffusionXLImg2ImgPipelinefrom diffusers.utils import load_imageimport osimport randomimport torchdef initialize_img2img(model_name, model_path, device, torch_dtype):    "initialize the img2img model"    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(        model_name, torch_dtype=torch_dtype, variant="fp16"    )    pipe = pipe.to(device)    if not (os.path.isdir(model_path)):        pipe.save_pretrained(model_path)    return pipedef gen_img2img(    pipe,    prompt,    device,    init_image_path="metadata/input_images/tmp_image.png",    image_name="output",    num_variations=1,    num_inference_steps=20,    guidance_scale=7.5,    height=512,    width=512,    strength = 0.5,    manual_seeds=None,    negative_prompt="",):    "generate an image"    if manual_seeds is None:        manual_seeds = [random.randint(0, 65000) for _ in range(num_variations)]    generator = [torch.Generator(device).manual_seed(i) for i in manual_seeds]        init_image = load_image(init_image_path).convert("RGB")    #init_image = init_image.resize(size=(512,512), resample=None)        images = pipe(        prompt=num_variations * [prompt],        num_inference_steps=num_inference_steps,        image=init_image,        guidance_scale=guidance_scale,        height=height,        width=width,        strength=strength,        generator=generator,        negative_prompt=num_variations * [negative_prompt],    ).images        img_paths = []    for i in range(len(images)):        img_path = f"metadata/output_images/{image_name}_seed_{manual_seeds[i]}.png"        images[i].save(img_path)        img_paths.append(img_path)    return img_paths