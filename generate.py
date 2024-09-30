import os
import json
import torch
import argparse
import imageio
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange, repeat
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from motionclone.models.unet import UNet3DConditionModel
from motionclone.pipelines.pipeline import MotionClonePipeline, GENONLY_MotionClonePipeline
from motionclone.pipelines.additional_components import customized_step, set_timesteps
from motionclone.utils.util import load_weights
from motionclone.utils.util import set_all_seed

# setup

prompts = [
    "Lion, moving its head, in the wild",
    "Bear, walking, on rocks",
    "Deer, grazing, in the forest",
    "Eagle, turning its head, on a branch",
    "Leopard, prowling, on grass",
    "Horse, trotting, in a field",
    "Penguin, waddling, on ice",
    "Elephant, swinging its trunk, in the savanna",
    "Panda, moving its head, in bamboo forest",
    "Kangaroo, hopping slowly, on dirt",
    "Zebra, walking, on grass",
    "Owl, rotating its head, in the forest",
    "Giraffe, lowering its head, in the savanna",
    "Fox, trotting, on grass",
    "Cat, stretching, on the couch",
    "Rabbit, nibbling, in the garden",
    "Dolphin, moving, near the surface of water",
    "Koala, moving its head, on a tree",
    "Camel, walking, in the desert",
    "Moose, walking, near a lake"
]
seed = 42

# torch.cuda.empty_cache()
# print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
# print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

config  = OmegaConf.load("configs/inference_config/generate.yaml")
model_config = OmegaConf.load(config.get("model_config", "configs.model_config.yaml"))

pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
motion_module_path = "models/Motion_Module/v3_sd15_mm.ckpt"
dreambooth_model_path = "models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors"
save_dir = "samples/"

warm_up_step = 10
cfg_scale = 7.5
num_inference_step = config.num_inference_step
adopted_dtype = torch.float16
device = "cuda"
config.width, config.height, config.video_length = 512, 512, 16

set_all_seed(seed)

# create validation pipeline
tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)
unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs)).to(device).to(dtype=adopted_dtype)
controlnet = None
unet.enable_xformers_memory_efficient_attention()

pipeline = GENONLY_MotionClonePipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
    controlnet=controlnet,
    scheduler = DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)),
).to(device)

pipeline = load_weights(
    pipeline,
    # motion module
    motion_module_path         = motion_module_path,
    dreambooth_model_path      = dreambooth_model_path,
).to(device)

pipeline.scheduler.customized_step = customized_step.__get__(pipeline.scheduler)
pipeline.scheduler.added_set_timesteps = set_timesteps.__get__(pipeline.scheduler)

generator = torch.Generator(device=pipeline.device)
generator.manual_seed(seed)
pipeline.scheduler.added_set_timesteps(num_inference_step, device=device)

video_name = prompts
for prompt in tqdm(prompts):

    prompt = prompt + ", 8k, high detailed, best quality, film grain, Fujifilm XT3, center-shot"
    config.new_prompt = prompt

    with torch.no_grad():
        videos = pipeline(
                        config = config,
                        generator = generator,
                        prompt=prompt
                    )
        videos = rearrange(videos, "b c f h w -> b f h w c")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,  config.new_prompt.replace(' ', '_') + f"GEN_seed{seed}_" + '.mp4')
    videos_uint8 = (videos[0] * 255).astype(np.uint8)
    imageio.mimwrite(save_path, videos_uint8, fps=8, quality=9)
    print(f"num_inference_steps: {num_inference_step}")
    print(f"saved on: {save_path}")

