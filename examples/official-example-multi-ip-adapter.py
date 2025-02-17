# see https://github.com/huggingface/diffusers/pull/6573

# yiyi testing script for multi-ipadapter: face + style folder
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1
)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", 
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
)

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"])
pipeline.set_ip_adapter_scale([0.7, 0.3])

pipeline.enable_model_cpu_offload()

face_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png")

style_folder = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy"
style_images =  [load_image(f"{style_folder}/img{i}.png") for i in range(10)]

generator = torch.Generator(device="cpu").manual_seed(0)

image = pipeline(
    prompt="wonderwoman",
    ip_adapter_image=[style_images, face_image],
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=50, num_images_per_prompt=1,
    generator=generator,
).images[0]
image.save(f"yiyi_test_out.png")