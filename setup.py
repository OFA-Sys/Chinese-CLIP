import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="cn_clip",
    py_modules=["cn_clip"],
    version="1.1",
    description="",
    author="OFA-Sys",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    data_files=[('clip/model_configs', ['cn_clip/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json',
                                        'cn_clip/clip/model_configs/RoBERTa-wwm-ext-large-chinese.json',
                                        'cn_clip/clip/model_configs/ViT-B-16.json',
                                        'cn_clip/clip/model_configs/ViT-B-32.json',
                                        'cn_clip/clip/model_configs/ViT-L-14.json']),
                ('clip/', ['cn_clip/clip/vocab.txt'])
                ],
    include_package_data=True,
)
