import numpy as np

class SSAORenderer:
    def __init__(self) -> None:
        # depth in world unit, in view space (x+ right, y+ down, z+ forward)
        self.m_view_depth : np.ndarray = None
        
        # normal in view space (x+ right, y+ down, z+ forward)
        self.m_view_normal : np.ndarray = None