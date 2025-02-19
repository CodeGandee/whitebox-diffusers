import torch
import numpy as np
from rich import print as pprint
import pyaigc.GlobalConfig as C
C.Logme.get_instance().set_level(C.Logme.Levels.DEBUG)

from test.sdxl_tester import SDXLTester, InputCase, OutputCase, SDTester, PipelineConfig
import igpy.myplot.jupyter_plot as jpt

# setup input files
input_cases = [
    InputCase(fn_image=r'sample-data/3c255ed3-4187-486e-8e2f-4f451852fd5d.png', positive_prompt='man portrait'),
    InputCase(fn_image=r'sample-data/7f51adf8-0d7d-4661-be1a-9a63b81c1f8f.png', positive_prompt='woman portrait'),
    InputCase(fn_image=r'sample-data/25c206f1-97bd-4de7-b770-9e8ae16085fd.png', positive_prompt='child portrait'),
]

# use your own model?
USE_CUSTOM_MODEL : bool = False

# use sdxl for sd 1.5?
USE_SDXL : bool = False

if USE_SDXL:
    from test.sdxl_tester import SDXLTester, InputCase, OutputCase
else:
    from test.sd_tester import SDTester, InputCase, OutputCase

# select model key, or specify custom model locations
if USE_SDXL:
    if USE_CUSTOM_MODEL:
        # use a model on your disk
        sd_key = C.SDXLModelKeys.CustomModel_1
        dir_sdxl_model = r'data/models/sdxl/sdxl-JuggernautXL-v5-diffusers'
        dir_canny_model = r'data/models/sdxl/canny'

        C.ModelConfigs.get_instance().set_base_model_path(sd_key, dir_sdxl_model)
        C.ModelConfigs.get_instance().set_base_model_path(C.ControlNetModelKeys.Canny_sdxl, dir_canny_model)
    else:
        # use default model paths
        sd_key = C.SDXLModelKeys.ProtoVision
else:
    if USE_CUSTOM_MODEL:
        # use a model on your disk
        sd_key = C.SDModelKeys.CustomModel_1
        dir_sd_model = r'data/models/diffusion/openjourney'
        dir_canny_model = r'data/models/controlnet/sd-controlnet-canny'

        C.ModelConfigs.get_instance().set_base_model_path(sd_key, dir_sd_model)
        C.ModelConfigs.get_instance().set_base_model_path(C.ControlNetModelKeys.Canny_sd_v15, dir_canny_model)
    else:
        # use default model paths
        sd_key = C.SDModelKeys.Dreamshaper_sd_v15

# where is the model?
model_info = C.ModelConfigs.get_instance().get_model_config(sd_key)
print('--- model components ---')
pprint(model_info)
print('------------------------')

# load model, create pipeline
if USE_SDXL:
    sdtest = SDXLTester()
else:
    sdtest = SDTester()
    
pc = PipelineConfig(
    sd_model_key=sd_key,
    with_canny_control=True
)
sdtest.load_model_pipeline(pc)

# img-to-img noise sequence
noise_endpoints = np.logspace(np.log10(600), np.log10(900), num=5, dtype=int)

# pick a case
ic = input_cases[1]
ic.batch_size = 2
ic.canny_weight = 0.3
ic.num_denoise_step = 10
ic.positive_prompt = "attractive young girl"
ic.negative_prompt = "toy, blur, naked"
long_edge_length = 640

# init
sdtest.init_input_case(ic)
ic.resize_by_long_edge(long_edge_length, divisible_by=8)
oc = sdtest.init_output_case(ic)

# denoise to the end
sdtest.denoise_dstate([oc.dstate])
img_batch = sdtest.model_pipeline.decode_latent_to_image(oc.dstate.latent, output_dtype=torch.uint8)

# get the image
imglist = [img_batch]
    
# add noise to dstate and denoise again
import copy
for noise_pt in noise_endpoints:
    _dstate = copy.deepcopy(oc.dstate)
    sdtest.add_noise_to_dstate(_dstate, to_timestep_index=noise_pt)
    sdtest.denoise_dstate([_dstate])
    img_batch = sdtest.model_pipeline.decode_latent_to_image(_dstate.latent, output_dtype=torch.uint8)
    imglist.append(img_batch)
    
# show images
print(noise_endpoints)
for idx_batch in range(ic.batch_size):
    imgs = [imglist[idx][idx_batch] for idx in range(len(imglist))]
    imgshow = np.column_stack(imgs)
    jpt.imshow(imgshow)