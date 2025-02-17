from typing import Self
from abc import ABC, abstractmethod

class IModel(ABC):
    ''' base interface or all models
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    @property
    @abstractmethod
    def model_key(self)->str:
        ''' a unique name of this type of model
        '''
        raise NotImplementedError()
    
    @property
    def config(self)->dict[str, object]:
        ''' arbitrary configuration info of this model
        '''
        raise NotImplementedError()
    
    @property
    def device(self) -> str | None:
        ''' device that host this model, following pytorch convention, 
        where None means "do not care" (maybe hosting on remote)
        '''
        return None
    
    # mutability control, required because the model is expected to be used by multiple pipelines
    @abstractmethod
    def set_mutable(self, is_mutable: bool):
        ''' set this model to be mutable or not
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def get_mutable(self) -> bool:
        ''' get if this model is mutable
        '''
        raise NotImplementedError()
    
    def clone(self, *args, **kwargs) -> Self:
        ''' clone this model into an independent instance.
        A cloned object is always mutable.
        '''
        import copy
        obj = copy.deepcopy(self)
        obj.set_mutable(True)
        return obj
    

class MutabilityMixin:
    ''' mixin for mutability control, implementing part of IModel interface,
    use it where you want to control the mutability of a model.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.m_is_mutable : bool = True
        
    def set_mutable(self, is_mutable: bool):
        self.m_is_mutable = is_mutable
        
    def get_mutable(self) -> bool:
        return self.m_is_mutable
    
    # decorator to check mutability
    def check_mutable(func):
        ''' decorator to check if the model is mutable, raise error if not.
        '''
        def wrapper(self, *args, **kwargs):
            if not self.get_mutable():
                raise ValueError(f'This {self.__class__.__name__} object is not mutable, if you did not set it, it is likely that the model is shared by others, please clone() it first.')
            return func(self, *args, **kwargs)
        return wrapper