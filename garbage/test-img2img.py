import torch
rng = torch.Generator()
# rng = torch.default_generator
import cv2
import numpy as np
from rich import print as pprint
import igpy.common.inout as cio
from tqdm.auto import tqdm

from docarray import BaseDoc
from docarray.typing import NdArray, TorchTensor

from pyaigc.pipeline.StableDiffusionWithControl import (
    DiffusionState,
    ControlnetData,
    ControlNetInfo,
    TextPromptData,
    StableDiffusionWithControl
)

import pyaigc.TestHelpers as th
import pyaigc.GlobalConfig as C

C.set_log_level(C.LoggingLevel.Info)
from pyaigc.loader.LocalModelLoader import LocalModelLoader

fn_test_image = r'sample-data/elijah-cobden-white-tiger-shot-hero.jpg'
simg = th.SampleImage.from_file(fn_test_image).resize_by_long_edge(long_edge=800, divisible_by=8)
prompt_text = "young snow white, beautiful, smile, medieval period, in city, masterpiece, high quality, very sharp, photorealistic"
prompt_text = ""
negative_text = 'bad face, blurry, low quality, multple person'

sd_key = C.SDModelKeys.Dreamshaper_sd_v15
comps = th.DiffusionPipelineComponent.from_sd_model_key(
    sd_model_key=sd_key, 
    with_vae=True,
    with_unet=True,
    with_canny_control=True)

if False:
    img = simg.as_uint8().get_image()
    x = comps.vae.image_to_encode_input(img)
    z = comps.vae.encode(x).latent_dist.mode()
    xx = comps.vae.decode(z).sample
    img2 = comps.vae.image_from_decode_output(xx)[0]
    img2 = th.SampleImage(img2).as_uint8().get_image()

    import igpy.myplot.jupyter_plot as jpt
    imgshow = np.column_stack([img, img2])
    jpt.imshow(imgshow)

obj = StableDiffusionWithControl()
obj.m_text_model = comps.text_model
obj.m_unet = comps.unet
obj.m_controlnets = {
    'canny': ControlNetInfo(model=comps.controlnet_canny, weight=1.0)
}
obj.m_scheduler = comps.scheduler
obj.m_vae = comps.vae

# target_device = obj.m_vae.device
# target_dtype = obj.m_vae.m_model.dtype


edgemap = simg.get_canny_edge()
h, w = edgemap.shape[:2]
dstate = DiffusionState(target_width=w, target_height=h)
dstate.controlnet_data['canny'] = ControlnetData(image_data=edgemap[None,..., None], 
                                                 image_data_layout='nhwc',
                                                 weight=0.5)
dstate.positive_prompt = TextPromptData(text=prompt_text)
dstate.negative_prompt = TextPromptData(text=negative_text)
# dstate.set_random_generator_state(rng.get_state(), rng.device)
# dstate.random_generator_state = rng.get_state()
num_inference_steps = 10
dstate.set_timestep_sequence_by_num_steps(num_inference_steps)


obj.init_diffusion_state(dstate)


img_input = simg.image
img_input = torch.from_numpy(img_input).unsqueeze(0).float() / 255.0
latent = obj.encode_image_to_latent(img_input)

noise = torch.randn_like(latent)
num_noise_add_steps = 1  #加噪声的步数，越大越不接近原图

# idx_t = np.argmin(np.abs(dstate.scheduler.m_all_timesteps - num_noise_add_steps))

# timestep_indices = np.linspace(0, idx_t, num_inference_steps, endpoint=True, dtype=int)[::-1]
timestep_indices = np.linspace(0, num_noise_add_steps, num_inference_steps, endpoint=True, dtype=int)[::-1]
dstate.set_timestep_indices(timestep_indices)
noisy_latents = dstate.scheduler.add_noise(latent, noise, num_noise_add_steps)

# t_noise = int(dstate.scheduler.m_all_timesteps[idx_t])
# t_noise = torch.tensor([t_noise], device=latent.device, dtype=torch.int64)
# noisy_latents = dstate.scheduler.scheduler.add_noise(latent, noise, t_noise)
dstate.latent = noisy_latents

while dstate.scheduler.next_step_index is not None:
    pipeline = obj
    pipeline.step_denoise([dstate])


# denoise from middle
if False:
    import copy
    dstate_middle : DiffusionState = copy.deepcopy(dstate)
    dstate_middle.set_current_step_index(middle_step_index)
    dstate_middle.latent = middle_latent
    for k in tqdm(range(middle_step_index, len(dstate.timestep_indices))):
        obj.step_denoise([dstate_middle])

img = obj.decode_latent_to_image(dstate.latent)[0]
# img_2 = obj.decode_latent_to_image(dstate_middle.latent)[0]
# x = obj.m_vae.decode(dstate.latent * 8).sample
# img = obj.m_vae.image_from_model_input(x)[0]
cv2.imwrite('test.png', img)
import igpy.myplot.jupyter_plot as jpt
jpt.imshow(img)
# jpt.imshow(img_2)