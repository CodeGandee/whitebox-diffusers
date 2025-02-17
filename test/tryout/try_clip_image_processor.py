import numpy as np
import torch
import logging
from igpy.common.util_logging import LoggingTemplates
import igpy.myplot.jupyter_plot as jpt
import sklearn.preprocessing as pp
logging.basicConfig(level=logging.INFO, format=LoggingTemplates.NameLevel)

from transformers import (
    CLIPImageProcessor
)

import igpy.common.inout as cio

fn_image = r'D:\code\aigc-gui\sample-data\3c255ed3-4187-486e-8e2f-4f451852fd5d.png'

img = cio.imread(fn_image)

proc = CLIPImageProcessor()
proc.preprocess(img)
from_np_hwc = proc(img, return_tensors='pt').pixel_values
logging.info(f'type(img)= {type(img).__name__}, img shape= {img.shape}, out shape= {from_np_hwc.shape}')

logging.info('show original image.')
jpt.imshow(img)

# test nparray
def tensor_to_ndarray(t : torch.Tensor) -> np.ndarray:
    ''' input t = tensor in NCHW format
    '''
    x : np.ndarray = t[0].permute(1,2,0).detach().cpu().numpy()
    z = pp.minmax_scale(x.reshape((-1,3)), feature_range=(0,1)).reshape(x.shape)
    return z
    

# imgshow = pp.minmax_scale(_imgshow.reshape((-1,3)), feature_range=(0,1)).reshape(_imgshow.shape)
# imgshow = tensor_to_ndarray(from_np_hwc)
# logging.info('recover image from tensor')
# jpt.imshow(imgshow)

# test tensor as input
# you need to call do_rescale=False to avoid 1.0/255.0 multiplication
logging.info('test float tensor as input')
img_tensor_nchw = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
from_tensor_nchw = proc(img_tensor_nchw, return_tensors='pt', do_rescale=False).pixel_values
imgshow = tensor_to_ndarray(from_tensor_nchw)
logging.info('recover image from tensor')
jpt.imshow(imgshow)

# tensor in hwc uint8 format, just call the processor like numpy
logging.info('test uint8 HWC tensor as input')
img_tensor_hwc = torch.from_numpy(img)
from_tensor_nchw = proc(img_tensor_hwc, return_tensors='pt').pixel_values
imgshow = tensor_to_ndarray(from_tensor_nchw)
logging.info('recover image from tensor')
jpt.imshow(imgshow)

# tensor in nchw uint8 format, just call the processor like numpy
logging.info('test uint8 NCHW tensor as input')
img_tensor_nchw = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
from_tensor_nchw = proc(img_tensor_nchw, return_tensors='pt').pixel_values
imgshow = tensor_to_ndarray(from_tensor_nchw)
logging.info('recover image from tensor')
jpt.imshow(imgshow)

# tensor in nhwc uint8 format, just call the processor like numpy
logging.info('test uint8 NHWC tensor as input')
img_tensor_nhwc = torch.from_numpy(img).unsqueeze(0)
from_tensor_nchw = proc(img_tensor_nhwc, return_tensors='pt').pixel_values
imgshow = tensor_to_ndarray(from_tensor_nchw)
logging.info('recover image from tensor')
jpt.imshow(imgshow)