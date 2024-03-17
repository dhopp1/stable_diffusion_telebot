import pandas as pd
import telebot
import os
import sys
import gc
import random
import torch

from helper.txt2img import initialize_txt2img, gen_txt2img
from helper.img2img import initialize_img2img, gen_img2img


# bot setup
BOT_TOKEN = sys.argv[1]

bot = telebot.TeleBot(BOT_TOKEN)

if not(os.path.isdir("metadata/models/")):
    os.mkdir("metadata/models/")
    
if not(os.path.isdir("metadata/input_images/")):
    os.mkdir("metadata/input_images/")
    
if not(os.path.isdir("metadata/output_images/")):
    os.mkdir("metadata/output_images/")

# List of user_id of authorized users
admins = list(pd.read_csv("metadata/admin_list.csv").user_id)

# list of models
model_list = pd.read_csv("metadata/model_list.csv")

# xl
if os.path.isdir(model_list.loc[0, "path"]):
    model_name = model_list.loc[0, "path"]
else:
    model_name = model_list.loc[0, "url"]

# outpainting
if os.path.isdir(model_list.loc[1, "path"]):
    outpainting_name = model_list.loc[1, "path"]
else:
    outpainting_name = model_list.loc[1, "url"]

# parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else None

num_variations = 1
num_inference_steps = 20
guidance_scale = 7.5
height = 512
width = 512
manual_seeds = None
negative_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature."
strength = 0.5


@bot.message_handler(commands=["start", "hello"])
def send_welcome(message):
    bot.send_message(
        message.chat.id, text="Initializing, this may take a few minutes..."
    )

    if message.from_user.id in admins:
        global txt2img_pipe
        global img2img_pipe

        # txt2img model
        txt2img_pipe = initialize_txt2img(
            model_name, model_list.loc[0, "path"], device, torch_dtype
        )
        
        # img2img model
        img2img_pipe = initialize_img2img(
            model_name, model_list.loc[0, "path"], device, torch_dtype
        )

        bot.reply_to(
            message,
            f"""Successfully initialized! You are generating with Stable Diffusion XL 1.0, information on prompting here: https://blog.segmind.com/prompt-guide-for-stable-diffusion-xl-crafting-textual-descriptions-for-image-generation/.
Default parameters are:
num_variations = {num_variations} (how many different variations on the same prompt to produce),
num_inference_steps = {num_inference_steps} (more = higher detail but longer generating times)
guidance_scale = {guidance_scale} (0-10, higher = more adhesion to prompt but maybe less variety)
height = {height} (height in pixels of output image)
width = {width} (width in pixels of output image)
manual_seeds = {manual_seeds} (list of integers as long as 'num_variations', in case you want to create the same image again)
negative_prompt = '{negative_prompt}' (what you don't want in the image')

To change these, pass the following as your prompt: '[params]{{'prompt':'prompt text', 'num_variations':2, 'manual_seeds':[4,7]}}', etc.
""",
        )
    else:
        bot.reply_to(message, "Unauthorized user")

# txt2img handler
@bot.message_handler(func=lambda msg: True, content_types = ["photo", "document", "text"])
def echo_all(message):
    bot.send_message(
        message.chat.id,
        text=f"generating...",
    )
    
    message_text = message.text if message.text is not None else message.caption

    # parameters in message
    if "[params]" in message_text:
        try:
            param_dict = eval(message_text.split("params]")[1])
        except:
            bot.send_message(message.chat.id, text="prompt format error, try again")
    else:
        param_dict = {"prompt": message_text}

    if True:
        # img2img
        if message.document is not None:
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            with open("metadata/input_images/tmp_image.png", 'wb') as new_file:
                new_file.write(downloaded_file)
                new_file.close()
                
            img_paths = gen_img2img(
                pipe=img2img_pipe,
                prompt=param_dict["prompt"],
                device=device,
                image_name=param_dict["image_name"]
                if "image_name" in param_dict.keys()
                else "output",
                num_variations=param_dict["num_variations"]
                if "num_variations" in param_dict.keys()
                else num_variations,
                num_inference_steps=param_dict["num_inference_steps"]
                if "num_inference_steps" in param_dict.keys()
                else num_inference_steps,
                guidance_scale=param_dict["guidance_scale"]
                if "guidance_scale" in param_dict.keys()
                else guidance_scale,
                height=param_dict["height"] if "height" in param_dict.keys() else height,
                width=param_dict["width"] if "width" in param_dict.keys() else width,
                strength=param_dict["strength"]
                if "strength" in param_dict.keys()
                else strength,
                manual_seeds=param_dict["manual_seeds"]
                if "manual_seeds" in param_dict.keys()
                else manual_seeds,
                negative_prompt=param_dict["negative_prompt"]
                if "negative_prompt" in param_dict.keys()
                else negative_prompt,
            )
                
        else:
            # txt2img
            img_paths = gen_txt2img(
                pipe=txt2img_pipe,
                prompt=param_dict["prompt"],
                device=device,
                image_name=param_dict["image_name"]
                if "image_name" in param_dict.keys()
                else "output",
                num_variations=param_dict["num_variations"]
                if "num_variations" in param_dict.keys()
                else num_variations,
                num_inference_steps=param_dict["num_inference_steps"]
                if "num_inference_steps" in param_dict.keys()
                else num_inference_steps,
                guidance_scale=param_dict["guidance_scale"]
                if "guidance_scale" in param_dict.keys()
                else guidance_scale,
                height=param_dict["height"] if "height" in param_dict.keys() else height,
                width=param_dict["width"] if "width" in param_dict.keys() else width,
                manual_seeds=param_dict["manual_seeds"]
                if "manual_seeds" in param_dict.keys()
                else manual_seeds,
                negative_prompt=param_dict["negative_prompt"]
                if "negative_prompt" in param_dict.keys()
                else negative_prompt,
            )
        # send the image
        for img_path in img_paths:
            bot.send_message(
                message.chat.id, text=f"{img_path.split('/')[-1].replace('.png', '')}: "
            )
            img = open(img_path, "rb")
            bot.send_photo(message.chat.id, img)
            img.close()
            os.remove(img_path)  # delete the image locally
        if os.path.exists("metadata/input_images/tmp_image.png"):
            os.remove("metadata/input_images/tmp_image.png")
    else:
        bot.send_message(message.chat.id, text="error encountered, please try again")


bot.infinity_polling()
