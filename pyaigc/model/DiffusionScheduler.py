# unified interface to the schedulers used in diffusion model
import numpy as np
import torch
import copy
from numpy.typing import NDArray

from typing import Any
from diffusers.schedulers import (
    SchedulerMixin,
    DPMSolverMultistepScheduler,    # used as a template
    UniPCMultistepScheduler,
    EulerDiscreteScheduler
)
from typing_extensions import Self

from pyaigc.GlobalConfig import SchedulerTypes as SchedulerTypeNames
from ismember import ismember

SupportedSchedulerType = DPMSolverMultistepScheduler | UniPCMultistepScheduler | EulerDiscreteScheduler

class IDiffusionScheduler:
    ''' wrapping diffuser schedulers into a unified interface
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    @classmethod
    def from_diffuser_scheduler(cls, scheduler : SupportedSchedulerType) -> Self:
        raise NotImplementedError()
    
    @classmethod
    def from_config(cls, config : dict, typename : str) -> Self:
        ''' create a scheduler from config and scheduler type
        
        parameters
        -----------
        config : dict
            config for the diffuser scheduler
        typename : str
            name of the scheduler type, must be one of the keys in SchedulerTypeNames
            
        return
        ---------
        scheduler : IDiffusionScheduler
            the constructed scheduler object
        '''
        raise NotImplementedError()
    
    def set_selected_timesteps_by_index(self, timestep_indices : np.ndarray):
        idx_ts = np.atleast_1d(timestep_indices)
        
        # modify scheduler to use the selected timesteps
        _timesteps = self.all_selectable_timesteps[idx_ts]
        _sigmas = self._append_extra_sigma(self.all_selectable_sigmas[idx_ts])
        _nstep = len(timestep_indices)
        
        sch = self.scheduler
        sch.set_timesteps(num_inference_steps=_nstep)   # just to clear internal state
        
        # change timesteps
        sch.timesteps = torch.tensor(_timesteps)
        sch.sigmas = torch.tensor(_sigmas, dtype=torch.float32)
        
    def set_current_step_index(self, step_index : int | None):
        ''' set current step index
        '''
        if step_index is None:
            self.scheduler._step_index = None
        else:
            self.scheduler._step_index = step_index + 1    # the scheduler stores the next step index
            
    def _generate_all_sigmas(self) -> tuple[np.ndarray, np.ndarray]:
        ''' return all timesteps and sigmas, the timesteps are in ascending order.
        
        return
        ----------
        timesteps : np.ndarray
            all timesteps
        sigmas : np.ndarray
            all sigmas, without the extra appended one
        '''
        # generate all sigmas
        sch_tmp = copy.deepcopy(self.scheduler)
        n_train_step = self.num_train_timesteps
        
        # overwrite the config
        mutable_config = dict(sch_tmp.config)
        # mutable_config['use_karras_sigmas'] = False
        # mutable_config['use_lu_lambdas'] = False
        mutable_config['timestep_spacing'] = 'linspace'
        sch_tmp._internal_dict = sch_tmp.config.__class__(mutable_config)
        
        # generate sigmas
        sch_tmp.set_timesteps(num_inference_steps=n_train_step)
        sigmas : np.ndarray = sch_tmp.sigmas.numpy()
        ts : np.ndarray = sch_tmp.timesteps.numpy()
        
        all_sigmas = np.ascontiguousarray(sigmas[:-1][::-1].copy())
        all_timesteps = np.ascontiguousarray(ts[::-1].copy())
        
        assert len(all_sigmas) == len(all_timesteps), 'length of sigmas and timesteps must be the same'
        
        return all_timesteps, all_sigmas
    
    def _append_extra_sigma(self, selected_sigmas : np.ndarray) -> np.ndarray:
        ''' append an extra sigma to the end of selected_sigmas, required by the diffuser,
        and different scheduler may have different requirements
        '''
        raise NotImplementedError()
    
    def add_noise(self, latent : torch.Tensor, 
                  noise : torch.Tensor, 
                  to_timestep_index : int,
                  from_timestep_index : int = -1, 
                  inplace : bool = False) -> torch.Tensor:
        ''' add noise to latent, from given noise level to target noise level, provided as time step index
        
        parameters
        -------------
        latent : torch.Tensor
            the latent of the image, to be denoised later
        noise : torch.Tensor
            the normal noise, zero mean and unit variance
        to_timestep_index : int
            the target timestep index, whose noise level will be reached after adding noise
        from_timestep_index : int
            the source timestep index, whose noise level will be used as the starting point
        '''
        assert isinstance(self.scheduler, DPMSolverMultistepScheduler) or \
            isinstance(self.scheduler, UniPCMultistepScheduler) or \
                isinstance(self.scheduler, EulerDiscreteScheduler), \
                    'scheduler must be DPMSolverMultistepScheduler or UniPCMultistepScheduler or EulerDiscreteScheduler'
                    
        assert latent.dim() == 4, 'latent must be 4D tensor in NCHW format'
        assert noise.dim() == 4, 'noise must be 4D tensor in NCHW format'
        assert to_timestep_index is not None, 'to_timestep_index must be specified'
        assert from_timestep_index is not None, 'from_timestep_index must be specified'
        # assert isinstance(to_timestep_index, int), 'to_timestep_index must be int'
        # assert isinstance(from_timestep_index, int), 'from_timestep_index must be int'
        
        # init_noise_sigma : float = getattr(self.scheduler, 'init_noise_sigma', 1.0)
            
        nbatch = latent.shape[0]
        idx_ts_to = np.atleast_1d([to_timestep_index] * nbatch).flatten().astype(int)
        idx_ts_from = np.atleast_1d([from_timestep_index] * nbatch).flatten().astype(int)
        
        out_latent = latent if inplace else latent.clone()
        for i in range(nbatch):
            _idx_t_to, _idx_t_from = idx_ts_to[i], idx_ts_from[i]
            sigma_to = self.all_selectable_sigmas[_idx_t_to]
            
            sigma_from : float | None = None
            if _idx_t_from >= 0:
                sigma_from = self.all_selectable_sigmas[_idx_t_from]
            
            if isinstance(self.scheduler, DPMSolverMultistepScheduler) or isinstance(self.scheduler, UniPCMultistepScheduler):
                alpha_t, sigma_t = self.scheduler._sigma_to_alpha_sigma_t(sigma_to)
                
                if sigma_from is None:
                    alpha_t_from = 1.0
                    sigma_t_from = 0.0
                else:
                    alpha_t_from, sigma_t_from = self.scheduler._sigma_to_alpha_sigma_t(sigma_from)
                    
                # rename variables for convenience
                # let latent_current = latent_noise_free * alpha_t_from + noise_1 * sigma_t_from
                # let latent_next = latent_current * alpha_t + noise_2 * sigma_t
                # so, latent_current ~ N(latent_noise_free * alpha_t_from, sigma_t_from**2)
                # latent_next ~ N(latent_noise_free * alpha_t, sigma_t**2)
                # the problem is how to transform latent_current to latent_next so that the distribution is correct
                # we first scale latent_current, and then add a scaled Gaussian noise to modify the variance
                
                # TODO: test this, add noise in the middle of the diffusion process
                a1, b1 = alpha_t_from, sigma_t_from
                a2, b2 = alpha_t, sigma_t
                alpha_final = a2/a1
                sigma_final = np.sqrt(np.clip(b2**2 - (b1*a2/a1)**2, 0, None))
                out_latent[i] = latent[i] * alpha_final + noise[i] * sigma_final
                
                # out_latent[i] = latent[i] * alpha_t + noise[i] * sigma_t
            elif isinstance(self.scheduler, EulerDiscreteScheduler):
                sigma_t_from : float = 0.0 if sigma_from is None else sigma_from
                b1, b2 =sigma_t_from, sigma_to
                
                # like the above, modify the variance by adding a scaled Gaussian noise
                sigma_final = np.sqrt(np.clip(b2**2-b1**2,0, None))
                out_latent[i] = latent[i] + noise[i] * sigma_final
            else:
                raise ValueError(f'invalid scheduler type: {type(self.scheduler)}')
        
        return out_latent
            
    @property
    def selected_timestep_indices(self) -> np.ndarray:
        ''' indices of selected timesteps
        '''
        ts = self.scheduler.timesteps
        all_ts = self.all_selectable_timesteps
        is_included, loc = ismember(ts, all_ts, 'elementwise')
        assert np.all(is_included), 'selected timesteps must be in all selectable timesteps'
        return loc
    
    @property
    def all_selectable_timesteps(self) -> np.ndarray:
        ''' all selectable timesteps, in as
        '''
        raise NotImplementedError()
    
    @property
    def all_selectable_sigmas(self) -> np.ndarray:
        ''' all selectable sigmas
        '''
        raise NotImplementedError()
    
    @property
    def scheduler(self) -> SupportedSchedulerType:
        raise NotImplementedError()
    
    @property
    def config(self) -> dict:
        return self.scheduler.config
    
    @property
    def num_train_timesteps(self) -> int:
        return self.scheduler.config.get('num_train_timesteps')
    
    @property
    def current_step_index(self) -> int | None:
        ''' current step index, None if not started yet
        '''
        if self.scheduler.step_index is None:
            return None
        else:
            return self.scheduler.step_index - 1
    
    @property
    def next_step_index(self) -> int | None:
        ''' next step index, None if denoising is finished
        '''
        sch = self.scheduler
        if len(sch.timesteps) == 0:
            return None
        
        t = self.current_step_index
        if t is None:
            return 0
        else:
            if t+1 < len(sch.timesteps):
                return t+1
            else:
                return None
    
    @property
    def current_timestep(self) -> int:
        ''' timestep already done
        '''
        if self.scheduler.step_index is not None:
            return self.scheduler.timesteps[self.scheduler.step_index-1]
        
    @property
    def next_timestep(self) -> int:
        ''' next timestep, None if denoising is finished
        '''
        idx_next_step = self.next_step_index
        if idx_next_step is None:
            return None
        else:
            return self.scheduler.timesteps[idx_next_step]
        
class DiffusionScheduler_EulerDescrete(IDiffusionScheduler):
    ''' wrapping diffuser schedulers into a unified interface
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.m_scheduler : EulerDiscreteScheduler = None
        
        # sigmas[i] is for timesteps[i], this is used for interpolation
        # self.m_idx_selected_timesteps : np.ndarray = None   # indices of selected timesteps
        self.m_all_timesteps : np.ndarray = None
        self.m_all_sigmas : np.ndarray = None

    @property
    def all_selectable_timesteps(self) -> np.ndarray:
        return self.m_all_timesteps
    
    @property
    def all_selectable_sigmas(self) -> np.ndarray:
        return self.m_all_sigmas
    
    @property
    def scheduler(self) -> EulerDiscreteScheduler:
        return self.m_scheduler
    
    @classmethod
    def from_diffuser_scheduler(cls, scheduler : EulerDiscreteScheduler) -> 'DiffusionScheduler_EulerDescrete':
        out = cls()
        out.m_scheduler = copy.deepcopy(scheduler)
        
        # call this to generate all sigmas
        out.m_scheduler.set_timesteps(num_inference_steps=scheduler.config.num_train_timesteps-1)
        
        # record sigmas
        ts, sigmas = out._generate_all_sigmas()
        out.m_all_timesteps = ts
        out.m_all_sigmas = sigmas
        
        return out
    
    @classmethod
    def from_config(cls, config : dict, 
                    typename : str = SchedulerTypeNames.EulerDiscrete) -> 'DiffusionScheduler_EulerDescrete':
        ''' create a scheduler from config and scheduler type
        
        parameters
        -----------
        config : dict
            config for the diffuser scheduler
        typename : str
            name of the scheduler type,
        '''
        import inspect
        
        # for dpm-related schedulers
        modelConfigKeys = inspect.signature(EulerDiscreteScheduler.__init__).parameters.keys() & config.keys()
        new_config = {k:config[k] for k in modelConfigKeys}
        sch : EulerDiscreteScheduler = EulerDiscreteScheduler.from_config(new_config)
        return cls.from_diffuser_scheduler(sch)
    
    def _append_extra_sigma(self, selected_sigmas: np.ndarray) -> np.ndarray:
        return np.concatenate([selected_sigmas, [0.0]]).astype(selected_sigmas.dtype)

class DiffusionScheduler(IDiffusionScheduler):
    ''' wrapping diffuser schedulers into a unified interface
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.m_scheduler : SupportedSchedulerType = None
        
        # sigmas[i] is for timesteps[i], this is used for interpolation
        # self.m_idx_selected_timesteps : np.ndarray = None   # indices of selected timesteps
        self.m_all_timesteps : np.ndarray = None
        self.m_all_sigmas : np.ndarray = None   
        
    @classmethod
    def from_diffuser_scheduler(cls, scheduler : EulerDiscreteScheduler) -> 'DiffusionScheduler_EulerDescrete':
        out = cls()
        out.m_scheduler = copy.deepcopy(scheduler)
        
        # call this to generate all sigmas
        out.m_scheduler.set_timesteps(num_inference_steps=scheduler.config.num_train_timesteps-1)
        
        # record sigmas
        ts, sigmas = out._generate_all_sigmas()
        out.m_all_timesteps = ts
        out.m_all_sigmas = sigmas
        
        return out
    
    @classmethod
    def from_config(cls, config : dict, typename : str) -> 'DiffusionScheduler':
        ''' create a scheduler from config and scheduler type
        
        parameters
        -----------
        config : dict
            config for the diffuser scheduler
        typename : str
            name of the scheduler type, must be one of the keys in SchedulerTypeNames
            
        return
        ---------
        scheduler : DiffusionScheduler
            the constructed scheduler object
        '''
        import inspect
        sch : SchedulerMixin = None
        
        # for dpm-related schedulers
        modelConfigKeys_dpm = inspect.signature(DPMSolverMultistepScheduler.__init__).parameters.keys() & config.keys()
        config_dpm = {k:config[k] for k in modelConfigKeys_dpm}
        
        if typename == SchedulerTypeNames.DPM_2M_Karras:
            config_dpm['use_karras_sigmas'] = True
            # sch = DPMSolverMultistepScheduler(**config_dpm)
        elif typename == SchedulerTypeNames.DPM_2M_SDE:
            config_dpm['use_karras_sigmas'] = False
            config_dpm['algorithm_type'] = 'sde-dpmsolver++'
            # sch = DPMSolverMultistepScheduler(**config_dpm)
        elif typename == SchedulerTypeNames.DPM_2M_SDE_Karras:
            config_dpm['use_karras_sigmas'] = True
            config_dpm['algorithm_type'] = 'sde-dpmsolver++'
            # sch = DPMSolverMultistepScheduler(**config_dpm)
        elif typename == SchedulerTypeNames.DPM_2M:
            # sch = DPMSolverMultistepScheduler(**config_dpm)
            pass
        sch = DPMSolverMultistepScheduler.from_config(config_dpm)
        
        # create unified scheduler
        out = cls.from_diffuser_scheduler(sch)
        return out
    
    def _append_extra_sigma(self, selected_sigmas: np.ndarray) -> np.ndarray:
        return np.concatenate([selected_sigmas, selected_sigmas[-1:]])
    
    @property
    def all_selectable_sigmas(self) -> np.ndarray:
        return self.m_all_sigmas
    
    @property
    def all_selectable_timesteps(self) -> np.ndarray:
        return self.m_all_timesteps
    
    @property
    def scheduler(self) -> SupportedSchedulerType:
        return self.m_scheduler
    
    @property
    def config(self) -> dict:
        return self.m_scheduler.config
    
    @property
    def max_timestep(self) -> int:
        return self.m_scheduler.config.num_train_timesteps
    
    @property
    def current_step_index(self) -> int | None:
        ''' current step index, None if not started yet
        '''
        if self.m_scheduler.step_index is None:
            return None
        else:
            return self.m_scheduler.step_index - 1
    
    @property
    def next_step_index(self) -> int | None:
        ''' next step index, None if denoising is finished
        '''
        sch = self.m_scheduler
        if len(sch.timesteps) == 0:
            return None
        
        t = self.current_step_index
        if t is None:
            return 0
        else:
            if t+1 < len(sch.timesteps):
                return t+1
            else:
                return None
    
    @property
    def current_timestep(self) -> int:
        ''' timestep already done
        '''
        if self.m_scheduler.step_index is not None:
            return self.m_scheduler.timesteps[self.m_scheduler.step_index-1]
        
    @property
    def next_timestep(self) -> int:
        ''' next timestep, None if denoising is finished
        '''
        idx_next_step = self.next_step_index
        if idx_next_step is None:
            return None
        else:
            return self.m_scheduler.timesteps[idx_next_step]