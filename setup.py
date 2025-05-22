from setuptools import find_packages, setup

setup(
    name="ssl-intrinsic-probing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "timm>=0.9.0",
        "hydra-core>=1.3.0",
        "wandb>=0.15.0",
    ],
    python_requires=">=3.8",
)
