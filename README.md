# stable_diffusion_telebot
Create a Telegram bot for image generation. Uses [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for txt2img and img2img, [ReV_Animated_Inpainting](https://huggingface.co/redstonehero/ReV_Animated_Inpainting) for outpainting.

# Setup
1. clone this repo
2. get a [Hugging Face](https://huggingface.co/docs/api-inference/en/quicktour) API token and set it to the environment variable `HF_TOKEN`. [Windows](https://phoenixnap.com/kb/windows-set-environment-variable) and [Mac](https://phoenixnap.com/kb/set-environment-variable-mac) instructions.
3. create a [Telegram](https://telegram.org/) account
4. search for the "BotFather" bot in telegram
5. once there, send `/newbot` in the chat and follow the instructions, record your new bot's API token
6. `pip install -r requirements.txt`
7. add allowed users' Telegram IDs to `metadata/admin_list.csv`. [Instructions](https://bigone.zendesk.com/hc/en-us/articles/360008014894-How-to-get-the-Telegram-user-ID) on how to get Telegram user ID.
8. run the bot from the command line with `python bot.py <bot API key>`. The Bot API key is from step 5.
9. go to Telegram, search for your bot and start chatting.
10. Prompting is very important, so read a [prompting guide](https://blog.segmind.com/prompt-guide-for-stable-diffusion-xl-crafting-textual-descriptions-for-image-generation/) for details.
11. To use img2img and outpainting, upload a .png file with the prompt in the captions.
12. To change hyperparameters, send your message in the form `[params]{'prompt':'prompt text', 'num_inference_steps':20}`, etc.
13. To use outpainting, upload the image and in the comment write e.g., `[params]{'prompt':'prompt text', 'outpainting':True, 'height':512, 'width':800}`. The original image will be centralized in the delimited canvas then outfilled. To outfill more in one direction, select an overall bigger size than you need then crop in.