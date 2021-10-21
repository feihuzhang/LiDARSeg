from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


setup(
    name='fusion',
    ext_modules=[
        CUDAExtension('fusion', [
            'src/fusion_cuda.cpp',
            'src/fusion_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
