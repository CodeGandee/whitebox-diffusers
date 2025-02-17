# compute SSAO given depth and normal map
import numpy as np
from igpy.datasyn.v1.UE_SceneRendering import UE_SceneRendering, UE_RenderedObject
from igpy.datasyn.v1.UE_StaticFrameLoader import UE_StaticFrameLoader

from igpy.datasyn.ExrImage import ExrImage
import igpy.myplot.jupyter_plot as jpt
import seaborn as sns
import igpy.myplot.vistaplot as vpt

MaxClipDistance = 20 * 100.0
hfov_deg = 61.927

data_dir = r'E:\OneDrive - 华兴科技公司\projects\renderAlg\ssao\Saved\MovieRenders'
fn_exr = data_dir + r'/mainseq.0004'
fn_camera = data_dir + r'/camera.json'
fn_scene = data_dir + r'/scene_structure.json'
exr = ExrImage()
exr.open_file(fn_exr)

# rd = UE_SceneRendering.create_from_exr_file(fn_exr_image=fn_exr, parse_now=True)
# alpha_map = rd.m_exr_image.get_channel('A')
# mask_foreground = alpha_map > 1-1e-3
# color_image = rd.get_color_image(dtype=np.float32, gamma=2.2, rgb_only=True)
