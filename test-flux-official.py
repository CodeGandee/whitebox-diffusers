import torch
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel

fn_model=r'data\models\flux\flux-fp8\flux1-dev-fp8.safetensors'

pipe = FluxPipeline.from_pretrained(fn_model, torch_dtype=torch.float8_e4m3fn)
pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("image.png")