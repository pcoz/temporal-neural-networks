"""
TNN - Temporal Neural Networks

A Dynamical Systems Approach to Stable and Robust Neural Computation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tnn",
    version="0.1.0",
    author="Edward Chalk",
    author_email="edward@fleetingswallow.com",
    description="Temporal Neural Networks: A Dynamical Systems Approach to Stable and Robust Neural Computation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pcoz/temporal-neural-networks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pandas>=1.3.0",
        ],
        "examples": [
            "pandas>=1.3.0",
            "matplotlib>=3.4.0",
        ],
    },
)
