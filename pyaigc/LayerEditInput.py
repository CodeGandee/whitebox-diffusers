# provides input to the per-layer editing interface
import numpy as np
import json
import re
from attrs import define, field, asdict
import attrs.converters as ac
import attrs.validators as av
from typing import Any, Callable, ClassVar
from typing_extensions import Self
import uuid

from pyaigc.TestHelpers import SampleImage

from igpy.datasyn.v1.UE_SceneRendering import UE_SceneRendering, ExrImage
import igpy.common.shortfunc as sf
import igpy.common.inout as cio

@define(kw_only=True)
class ObjectMaskDesc:
    object_name : str = field(validator=av.instance_of(str))
    mask_image : str = field(validator=av.instance_of(str))

@define(kw_only=True)
class LayerInputFileDesc:
    scene_name : str = field(factory=lambda : f'scene.{uuid.uuid4().hex}',
                             validator=av.instance_of(str))
    
    color_image : str = field(default='color.png')
    depth_image : str = field(default='depth.png')
    canny_image : str = field(default='canny.png')
    object_list : list[ObjectMaskDesc] = field(factory=list)

@define(kw_only=True)
class LayerEditInput:
    InvalidDepthValue : ClassVar[float] = 0.0
    DepthPreferRange : ClassVar[tuple[float, float]] = (1e-3, 1e4)  #(min, max)
    
    # floating point image range in (0,1), must exists
    m_color_data : np.ndarray = field(alias='color_data', validator=av.instance_of(np.ndarray))
    
    # alpha channel, if None, assuming all ones
    m_alpha_data : np.ndarray | None = field(alias='alpha_data', default=None, 
                                             validator=av.optional(av.instance_of(np.ndarray)))
    
    # original depth data
    m_depth_data : np.ndarray = field(alias='depth_data', default=None, 
                                      validator=av.instance_of(np.ndarray))
    
    # process depth data before converting to uint8 or uint16 image
    m_depth_preprocessor : Callable[[np.ndarray], np.ndarray] | None = \
        field(alias='depth_preprocessor')
    
    # binary masks for objects
    m_name2mask : dict[str, np.ndarray] | None = field(alias='name2mask', default=None)
    
    # canny edge parameters
    m_canny_thres_1 : float = field(alias='canny_thres_1', default=100.0, init=False)
    m_canny_thres_2 : float = field(alias='canny_thres_2', default=200.0, init=False)
    
    def __attrs_post_init__(self):
        if self.m_alpha_data is None:
            self.m_alpha_data = np.ones(self.m_color_data.shape[:2], dtype=np.float32)
            
    @m_depth_preprocessor.default
    def __default_depth_preprocessor(self):
        return lambda x : np.log(x.clip(*self.DepthPreferRange))
    
    @m_color_data.validator
    def __validate_color_data(self, attribute, value : np.ndarray):
        assert len(value.shape) == 3, 'color data must be 3 channels'
        assert value.dtype in (np.float32, np.float64), 'color data must be floating point'
        assert value.min()>=0 and value.max()<=1, 'color data must be in range (0,1)'
        
    @m_alpha_data.validator
    def __validate_alpha_data(self, attribute, value : np.ndarray):
        if value is not None:
            assert len(value.shape) == 2, 'alpha data must be 2d'
            assert np.allclose(value.shape, self.m_color_data.shape[:2]), 'alpha data must have the same shape as color data'
        
    @m_depth_data.validator
    def __validate_depth_data(self, attribute, value : np.ndarray):
        assert value is not None, 'depth data must be provided'
        assert len(value.shape) == 2, 'depth data must be 2d'
        assert np.allclose(value.shape, self.m_color_data.shape[:2]), 'depth data must have the same shape as color data'
            
    @m_name2mask.validator
    def __validate_name2mask(self, attribute, value : dict[str, np.ndarray]):
        if value is not None:
            for k, v in value.items():
                assert len(v.shape) == 2, 'mask must be 2d'
                assert np.allclose(v.shape, self.m_color_data.shape[:2]), 'mask must have the same shape as color data'
    
    @classmethod
    def from_unreal_rendering(cls, ue_render : UE_SceneRendering, 
                       names_ignore : list[str] = None) -> Self:
        ''' create LayerEditInput from ExrImage
        '''
        name2obj = ue_render.rendered_objects_by_name
        _rgba : np.ndarray = ue_render.get_color_image(dtype=np.float32, gamma=2.2)
        _color_data : np.ndarray = np.ascontiguousarray(_rgba[:,:,:3]).clip(0,1)
        _alpha_data : np.ndarray = np.ascontiguousarray(_rgba[:,:,-1])
        _depth_data : np.ndarray = ue_render.get_depth_image(dtype=np.float32)
        # depth_channel = r'FinalImageMovieRenderQueue_WorldDepth.R'
        # _depth_data = ue_render.m_exr_image.get_channel(depth_channel).astype(np.float32)
        
        _name2mask : dict[str, np.ndarray] = {}
        
        for name, obj in name2obj.items():
            if names_ignore is not None and name in names_ignore:
                continue
            if obj.pixel_indices is not None and len(obj.pixel_indices) > 0:
                _name2mask[name] = obj.mask
            
        return cls(
            color_data = _color_data,
            depth_data = _depth_data,
            alpha_data = _alpha_data,
            name2mask = _name2mask
        )
        
    def merge_mask(self, from_names : list[str], to_name : str):
        ''' merge some masks into one
        '''
        for name in from_names:
            assert name in self.m_name2mask, f'{name} not in name2mask'
            
        mask = np.zeros_like(self.m_name2mask[from_names[0]]).astype(float)
        for name in from_names:
            mask += self.m_name2mask[name]
        
        # remove merged masks
        for name in from_names:
            del self.m_name2mask[name]
            
        # add new mask
        self.m_name2mask[to_name] = mask > 0
        
    def get_color_image(self, dtype = np.uint8 | np.uint16) -> np.ndarray:
        img = self.m_color_data
        val_max = np.iinfo(dtype).max
        img = (np.clip(img, 0, 1) * val_max).astype(dtype)
        return img
    
    def get_depth_image(self, dtype = np.uint8 | np.uint16, 
                        depth_data : np.ndarray = None) -> np.ndarray:
        ''' get depth image
        
        parameters
        ------------
        dtype : np.dtype
            output data type, can be np.uint8 or np.uint16
        depth_data : np.ndarray
            depth data to use, if None, use self.m_depth_data,
            this allows you to apply any preprocess to depth_data before passing it to this function.
            
        return
        ---------
        output : np.ndarray
            depth image, closer is brighter
        '''
        if depth_data is None:
            dp : np.ndarray = self.m_depth_data.copy()
        else:
            dp = depth_data.copy()
            
        # convert to int
        if self.m_depth_preprocessor is not None:
            dp = self.m_depth_preprocessor(dp)
            
        dp[self.m_alpha_data==0] = self.InvalidDepthValue
        validmask = dp != self.InvalidDepthValue
        valid_values = dp[validmask]
        normalize_values = (valid_values - valid_values.min()) / (valid_values.max() - valid_values.min())
        dp[validmask] = (1 - normalize_values) * np.iinfo(dtype).max
        dp = dp.astype(dtype)
        
        return dp
    
    def get_canny_edge_image(self, thres_1:float=None, 
                             thres_2:float=None) -> np.ndarray:
        ''' get canny edge image
        
        parameters
        --------------
        thres_1, thres_2 : float
            threshold for canny edge detection, if None, use self.m_canny_thres_1 and self.m_canny_thres_2
            
        return
        ----------
        output : np.ndarray
            canny edge image, in uint8 format
        '''
        if thres_1 is None:
            thres_1 = self.m_canny_thres_1
        if thres_2 is None:
            thres_2 = self.m_canny_thres_2
            
        img = self.get_color_image(dtype=np.uint8)
        canny = SampleImage.from_ndarray(img).get_canny_edge(thres_1 = thres_1, thres_2 = thres_2)
        canny[canny > 0] = 255
        return canny
        
    def export_to_dir(self, dir_output : str, 
                      depth_clip_min_max : tuple[float, float] = None,
                      verbose : bool = False):
        ''' export to a directory
        '''
        import os
        import json
        import cv2
        
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)
        
        desc = LayerInputFileDesc()    
        
        # save color image
        fn_out_color = f'{dir_output}/{desc.color_image}'
        img_color = self.get_color_image(dtype=np.uint8)
        if verbose:
            print(f'save color image to {fn_out_color}')
        cio.imwrite(fn_out_color, img_color)
        
        # save depth image
        fn_out_depth = f'{dir_output}/{desc.depth_image}'
        img_depth = self.get_depth_image(dtype=np.uint8)
        if verbose:
            print(f'save depth image to {fn_out_depth}')
        cio.imwrite(fn_out_depth, img_depth)
        
        # save canny
        fn_out_canny = f'{dir_output}/{desc.canny_image}'
        img_canny = self.get_canny_edge_image()
        if verbose:
            print(f'save canny image to {fn_out_canny}')
        cio.imwrite(fn_out_canny, img_canny)
        
        # save masks
        for name, mask in self.m_name2mask.items():
            fn_out_mask = f'{dir_output}/mask_{name}.png'
            desc.object_list.append(ObjectMaskDesc(
                object_name = name,
                mask_image = os.path.basename(fn_out_mask)
            ))
            if verbose:
                print(f'save mask {name} to {fn_out_mask}')
            cio.imwrite(fn_out_mask, (mask>0).astype(np.uint8) * 255)
            
        # save description
        fn_json = f'{dir_output}/info.json'
        if verbose:
            print(f'save description to {fn_json}')
        with open(fn_json, 'w+') as f:
            json.dump(asdict(desc), f, indent=4)
        
        if verbose:
            print('done')
        
