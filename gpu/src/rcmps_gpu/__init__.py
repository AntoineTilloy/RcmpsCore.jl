"""GPU implementations of RcmpsCore: the CUDA ground-state optimizer (rcmps_gpu.optimizer)
and the tangent-space excitation solver (rcmps_gpu.excitations).

Importing the package, or either subpackage, imports neither CuPy nor any submodule;
CuPy is needed only to run on a GPU.
"""

__version__ = "0.1.0"
