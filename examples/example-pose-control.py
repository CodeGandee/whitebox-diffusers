import os

# download models if needed
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '0'

import torch
import numpy as np
from rich import print as pprint
import pyaigc.GlobalConfig as C
from controlnet_aux import OpenposeDetector, ZoeDetector
C.Logme.get_instance().set_level(C.Logme.Levels.DEBUG)

from test.sdxl_tester import SDXLTester, InputCase, OutputCase, SDTester, PipelineConfig
import igpy.myplot.jupyter_plot as jpt
import igpy.common.image_processing as ip

# setup input files
input_cases = [
    InputCase(fn_image=r'sample-data/a1ee6e05-ee42-4883-96c0-21f1d86a5ce4.png', positive_prompt='woman portrait'),
    InputCase(fn_image=r'sample-data/69a344c8-7765-4d44-b6c3-4be9576da8a5.png', positive_prompt='woman dance'),
]

# use your own model?
USE_CUSTOM_MODEL : bool = False

# use sdxl for sd 1.5?
USE_SDXL : bool = True

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

if False:
    # if you want to use offline model, download annotators, you need the body_pose_model.pth
    # https://hf-mirror.com/lllyasviel/Annotators/tree/main
    dir_annotators = r'data/models/Annotators'
    openpose = OpenposeDetector.from_pretrained(dir_annotators)
else:
    # use online model
    openpose = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
# zoedepth = ZoeDetector.from_pretrained(dir_annotators)

# load model, create pipeline
if USE_SDXL:
    sdtest = SDXLTester()
else:
    sdtest = SDTester()
    
pc = PipelineConfig(
    sd_model_key=sd_key,
    with_openpose_control=True
)
sdtest.load_model_pipeline(pc)

# img-to-img noise sequence
noise_endpoints = np.logspace(np.log10(600), np.log10(900), num=3, dtype=int)

# pick a case
for ith_case, ic in enumerate(input_cases):
    ic.batch_size = 2
    # ic.canny_weight = 0.3
    ic.openpose_weight = 0.8
    ic.num_denoise_step = 20
    ic.positive_prompt = "attractive young girl"
    ic.negative_prompt = "toy, blur, naked"
    long_edge_length = 640

    # init
    sdtest.init_input_case(ic)

    # get openpose image
    img = ic.image
    openpose_image = np.array(openpose(img))
    openpose_image = ip.imresize(openpose_image, (img.shape[1], img.shape[0]))
    ic.openpose_image = openpose_image

    ic.resize_by_long_edge(long_edge_length, divisible_by=8)
    oc = sdtest.init_output_case(ic)

    # assert False

    # denoise to the end
    sdtest.denoise_dstate([oc.dstate])
    img_batch = sdtest.model_pipeline.decode_latent_to_image(oc.dstate.latent, output_dtype=torch.uint8)

    for idx_batch in range(ic.batch_size):
        imgshow = img_batch[idx_batch]
        imgshow = np.concatenate([ic.image, ic.openpose_image, imgshow], axis=1)
        jpt.imshow(imgshow)