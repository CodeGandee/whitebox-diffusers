import torch
import numpy as np
from rich import print as pprint
import copy
import attrs
from attrs import define, field, asdict
import cv2
from tqdm.auto import tqdm

import igpy.myplot.jupyter_plot as jpt
import igpy.common.inout as cio
import igpy.common.shortfunc as sf
import igpy.common.image_processing as ip

import pyaigc.GlobalConfig as C
import pyaigc.TestHelpers as th
from pyaigc.loader.LocalModelLoader import LocalModelLoader

from test.sd_local_edit_tester import (
    LocalEditInputCase, MaskedDenoiseOutputCase,
    SDTester, LocalEditTester, ImageBlender
)
from test.sdxl_tester import (
    SDXLTester, SDXLDiffusionState, PipelineConfig
)

C.Logme.get_instance().set_level(C.Logme.Levels.DEBUG)

# input cases
caselist : list[LocalEditInputCase] = [
    LocalEditInputCase(
        fn_image=r'sample-data/39b09e0d-4d22-4270-931d-a537bf2f449e.png',
        fn_inner_mask=r'sample-data\mask-edit\39b09e0d-4d22-4270-931d-a537bf2f449e\inner-mask.png',
        # outer_bbox_xywh=np.array([104, 280, (495-104+1)//8*8, (662-280+1)//8*8], dtype=int)
        outer_bbox_xywh=np.array([104, 280, 495-104+1, 662-280+1], dtype=int)
    )
]

LocalModelLoader.get_instance().m_cache_config.cache_unet = True

# use your own model?
USE_CUSTOM_MODEL : bool = False

# use sdxl for sd 1.5?
USE_SDXL : bool = False

# visualize denoise process
USE_PER_STEP_VISUALIZATION : bool = False

# select model key, or specify custom model locations
if USE_SDXL:
    if USE_CUSTOM_MODEL:
        # use a model on your disk
        sd_model_key = C.SDXLModelKeys.CustomModel_1
        dir_sdxl_model = r'data/models/sdxl/sdxl-JuggernautXL-v5-diffusers'
        dir_canny_model = r'data/models/sdxl/canny'

        C.ModelConfigs.get_instance().set_base_model_path(sd_model_key, dir_sdxl_model)
        C.ModelConfigs.get_instance().set_base_model_path(C.ControlNetModelKeys.Canny_sdxl, dir_canny_model)
    else:
        # use default model paths
        sd_model_key = C.SDXLModelKeys.ProtoVision
else:
    if USE_CUSTOM_MODEL:
        # use a model on your disk
        sd_model_key = C.SDModelKeys.CustomModel_1
        dir_sd_model = r'data/models/diffusion/openjourney'
        dir_canny_model = r'data/models/controlnet/sd-controlnet-canny'

        C.ModelConfigs.get_instance().set_base_model_path(sd_model_key, dir_sd_model)
        C.ModelConfigs.get_instance().set_base_model_path(C.ControlNetModelKeys.Canny_sd_v15, dir_canny_model)
    else:
        # use default model paths
        sd_model_key = C.SDModelKeys.Dreamshaper_sd_v15

# setup pipeline
if USE_SDXL:
    sd_test_helper = SDXLTester()
else:
    sd_test_helper = SDTester()
    
pc = PipelineConfig(
    sd_model_key=sd_model_key,
    with_canny_control=True
)

sd_test_helper.load_model_pipeline(pc)
local_edit_helper = LocalEditTester(sd_tester=sd_test_helper)

# pick a case and configure it
ic = caselist[0]
ic.batch_size = 1
ic.canny_weight = 0.5
ic.num_denoise_step = 20
ic.outer_bbox_scaled_long_edge = 600
ic.noise_timestep_indices = np.logspace(np.log10(300), np.log10(999), num=9, dtype=int)
ic.positive_prompt = 'flowers holding in the hand'
local_edit_helper.init_input_case(ic)

# create regional denoise output case
oc_region = local_edit_helper.make_masked_output_case(ic)
dstate = oc_region.dstate
if dstate.latent_denoise_mask is not None:
    thres_mask = torch.tensor(dstate.latent_denoise_mask > 0.5, dtype=dstate.latent.dtype)
    dstate.latent_denoise_mask = thres_mask
    
    
RepeatTimes = 10
NoiseLevel = 900

# add noise
noise = torch.randn_like(oc_region.dstate.latent)
dstate = oc_region.dstate
output_images = []
for i in range(RepeatTimes):
    print(f'iteration {i}')
    sd_test_helper.add_noise_to_dstate(dstate, to_timestep_index=NoiseLevel)
    sd_test_helper.denoise_dstate([dstate])
    denoised_image = sd_test_helper.model_pipeline.decode_latent_to_image(dstate.latent, output_dtype=torch.uint8)[0]
    output_images.append(denoised_image)
    jpt.imshow(denoised_image)
    
assert False

# use clean latent + noise for blending
# TODO: result is not nice, may have to do with linear combination of Gaussians, variance changes unexpectedly
# oc_region.dstate.latent_clean = oc_region.dstate.latent.clone()

# add noise and denoise region, for each batch
region_output_img_per_batch : list[list[np.ndarray]] = [[] for _ in range(ic.batch_size)]
region_blenders : list[ImageBlender] = []
noise = torch.randn(oc_region.dstate.latent.shape, generator=oc_region.dstate.random_generator)
for noise_pt in ic.noise_timestep_indices:
    
    # add noise
    sd_test_helper.add_noise_to_dstate(oc_region.dstate, 
                                       to_timestep_index=noise_pt, 
                                       noise=noise)
    
    if USE_PER_STEP_VISUALIZATION:
        # denoise and show progress
        should_continue = True
        while should_continue:
            should_continue = sd_test_helper.denoise_dstate([oc_region.dstate], num_step=1, noise=noise)
            
            # show it
            oc_current = sd_test_helper.model_pipeline.decode_latent_to_image(oc_region.dstate.latent)
            jpt.imshow(oc_current[0])
    else:
        sd_test_helper.denoise_dstate([oc_region.dstate])
        
    img_blender = local_edit_helper.init_image_blending(oc_region)
    region_blenders.append(img_blender)
    
    img_merge = img_blender.get_fullsize_image_by_poisson_blending()
    for ith_batch, _img in enumerate(img_merge):
        region_output_img_per_batch[ith_batch].append(_img)
        
jpt.imshow(oc_region.input_case_fullsize.image)

n_image_per_row = np.sqrt(len(region_output_img_per_batch[0])).astype(int)
for ith_batch, imgs in enumerate(region_output_img_per_batch):
    imgshow_list = sf.concat_images_to_groups(imgs, n_image_per_row)
    for x in imgshow_list:
        jpt.imshow(x)
        
