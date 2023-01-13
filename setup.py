import os

import pkg_resources
from setuptools import setup, find_packages

packages = find_packages(exclude=["tests*"])
with open('README_En.md', 'r', encoding='utf-8') as fp:
    long_description = fp.read()
setup(
    name="cn_clip",
    py_modules=["cn_clip"],
    version="1.5.1",
    author="OFA-Sys",
    author_email="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages = packages,
    keywords='clip',
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
                                        'cn_clip/clip/model_configs/ViT-L-14.json',
                                        'cn_clip/clip/model_configs/ViT-L-14-336.json',
                                        'cn_clip/clip/model_configs/ViT-H-14.json',
                                        'cn_clip/clip/model_configs/RN50.json',
                                        'cn_clip/clip/model_configs/RBT3-chinese.json'
                                        ]),
                ('clip/', ['cn_clip/clip/vocab.txt'])
                ],
    include_package_data=True,
    url='https://github.com/OFA-Sys/Chinese-CLIP',
    description='the Chinese version of CLIP.'
)
