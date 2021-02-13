from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='eppcoreops',
    ext_modules=[
        CUDAExtension('eppcoreops', [
            'eppcore_cpp.cpp',
            'eppcore_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })