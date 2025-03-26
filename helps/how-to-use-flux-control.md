
# How to Use XLabs-AI's Flux ControlNets in Python

First, let's look at how to use the XLabs-AI Flux ControlNets with the diffusers library. The `FluxControlNetPipeline` is now supported in diffusers (as of PR #9687).

## Basic Usage Example

This example demonstrates using a single Canny ControlNet:

```python
import torch
from diffusers import FluxControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from controlnet_aux import CannyDetector

# 1. Load the ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "XLabs-AI/flux-controlnet-canny-diffusers",
    torch_dtype=torch.float16
)

# 2. Create the FluxControlNetPipeline
pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 3. Prepare your control image (example with Canny edge detection)
input_image = load_image("your_input_image.jpg")
canny_detector = CannyDetector()
control_image = canny_detector(input_image, low_threshold=100, high_threshold=200)

# 4. Generate the controlled image
result = pipeline(
    prompt="your prompt here, high quality, detailed",
    negative_prompt="low quality, worst quality, bad anatomy",
    image=control_image,
    num_inference_steps=25,
    guidance_scale=4.0,
    controlnet_conditioning_scale=0.8,  # Adjust strength of control
    height=1024,
    width=1024
).images[0]

result.save("output.png")
```

## Using Multiple ControlNets Together

This example demonstrates using both Canny and Depth ControlNets simultaneously:

```python
import torch
from diffusers import FluxControlNetPipeline, ControlNetModel, MultiControlNetModel
from diffusers.utils import load_image
from controlnet_aux import CannyDetector, MidasDetector

# 1. Load the individual ControlNet models
canny_controlnet = ControlNetModel.from_pretrained(
    "XLabs-AI/flux-controlnet-canny-diffusers",
    torch_dtype=torch.float16
)

depth_controlnet = ControlNetModel.from_pretrained(
    "XLabs-AI/flux-controlnet-depth-diffusers",
    torch_dtype=torch.float16
)

# 2. Create a MultiControlNetModel
controlnet = MultiControlNetModel([canny_controlnet, depth_controlnet])

# 3. Create the pipeline
pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

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
```

## Memory Optimization for 16GB VRAM

To use these models on a system with only 16GB VRAM:

```python
import torch
from diffusers import FluxControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from controlnet_aux import CannyDetector

# Load model with bfloat16 precision 
controlnet = ControlNetModel.from_pretrained(
    "XLabs-AI/flux-controlnet-canny-diffusers",
    torch_dtype=torch.bfloat16
)

pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
)

# Enable CPU offloading (crucial for memory efficiency)
pipeline.enable_model_cpu_offload()

# Rest of the code remains the same
```

## Parameter Recommendations

Based on the information from official sources:

- **Canny ControlNet**:
  - `controlnet_conditioning_scale`: 0.5-0.8
  - `low_threshold`: 100 (adjust lower for more edges)
  - `high_threshold`: 200 (adjust higher for cleaner edges)

- **Depth ControlNet**:
  - `controlnet_conditioning_scale`: 0.6-0.9
  - Better for maintaining spatial composition

- **HED ControlNet**:
  - `controlnet_conditioning_scale`: 0.6-0.8
  - Good for copying general shapes while allowing creative freedom

These examples should give you a solid foundation for using XLabs-AI's Flux ControlNets with diffusers in Python.

[Source: huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_flux](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_flux)
[Source: github.com/XLabs-AI/x-flux](https://github.com/XLabs-AI/x-flux)
