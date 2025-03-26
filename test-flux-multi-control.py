import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"

import torch
from diffusers import (
    FluxControlNetPipeline, 
    FluxControlNetModel, 
    FluxMultiControlNetModel,
)
from diffusers.utils import load_image
from controlnet_aux import CannyDetector, MidasDetector

dir_canny_controlnet = r"data/models/flux/flux-controlnet-canny-diffusers"
dir_depth_controlnet = r"data/models/flux/flux-controlnet-depth-diffusers"
dir_flux_model = r'data/models/flux/FLUX.1-schnell'

# 1. Load the individual ControlNet models
canny_controlnet = FluxControlNetModel.from_pretrained(
    dir_canny_controlnet,
    torch_dtype=torch.float16
)

depth_controlnet = FluxControlNetModel.from_pretrained(
    dir_depth_controlnet,
    torch_dtype=torch.float16
)

# 2. Create a MultiControlNetModel
controlnet = FluxMultiControlNetModel([canny_controlnet, depth_controlnet])

# 3. Create the pipeline
pipeline = FluxControlNetPipeline.from_pretrained(
    dir_flux_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

pipeline.enable_model_cpu_offload()

assert False

# Enable CPU offloading to reduce VRAM usage
pipeline.enable_model_cpu_offload()

# 4. Prepare control images
input_image = load_image("your_input_image.jpg")

# Process for Canny
canny_detector = CannyDetector()
canny_image = canny_detector(input_image, low_threshold=100, high_threshold=200)

# Process for Depth
depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
depth_image = depth_detector(input_image)

# 5. Generate with multiple control images
result = pipeline(
    prompt="your prompt here, high quality, detailed",
    negative_prompt="low quality, worst quality, bad anatomy",
    image=[canny_image, depth_image],
    num_inference_steps=25,
    guidance_scale=4.0,
    controlnet_conditioning_scale=[0.6, 0.8],  # Different weights for each control
    height=1024,
    width=1024
).images[0]

result.save("multi_control_output.png")