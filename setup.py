from setuptools import setup, find_packages

setup(
    name="nys-newton",
    version="1.0.0",
    author="Your Name",
    author_email="hardiktankaria1406@gmail.com",
    description="PyTorch implementation of Nys-Newton optimizer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hardy-opt/Nys-Opt",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "black", "flake8"],
        "notebooks": ["jupyter>=1.0.0", "matplotlib>=3.3.0"],
    },
)