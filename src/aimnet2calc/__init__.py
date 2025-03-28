__version__ = '1.0.0'

__all__ = ['AIMNet2Calculator']


from .calculator import AIMNet2Calculator

try:
    from .aimnet2ase import AIMNet2ASE
    __all__.append('AIMNet2ASE')
except ImportError:
    import warnings
    warnings.warn('ASE is not installed. AIMNet2ASE will not be available.')
    pass

try:
    from .aimnet2pysis import AIMNet2Pysis
    __all__.append('AIMNet2Pysis')
except ImportError:
    import warnings
    warnings.warn('PySisiphus is not installed. AIMNet2Pysis will not be available.')
    pass