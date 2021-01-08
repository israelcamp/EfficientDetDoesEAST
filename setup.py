from setuptools import find_packages, setup

setup(
    name='effdet',
    packages=find_packages(exclude=["notebooks"]),
    version='0.1.0',
    description='EfficientDet',
    author='Israel',
    license='MIT',
    install_requires=[
        'torch>1.5',
        'efficientnet_pytorch>=0.7.0',
        'numpy>=1'
    ]
)
