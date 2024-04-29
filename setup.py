from setuptools import setup, find_packages

from lang_efficient_sam import __version__

setup(
    name="lang-efficient-sam",
    version=__version__,

    url="https://github.com/Aedelon/Lang-Efficient-SAM",
    author="Delanoe Pirard",
    author_email="delanoe.pirard.pro@gmail.com",

    py_modules=find_packages(),

    install_requires=[
        'torch',
        'groundingdino-py',
        'numpy',
        'torchvision'
    ]
)