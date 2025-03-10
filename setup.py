from setuptools import setup, find_packages

setup(
    name="gameoflife_cuda",  # PyPI package name
    version="0.1.0",  # Increment for future updates
    packages=find_packages(),  # Automatically find all packages
    author="Sagnik Nath",
    author_email="sanath@ucsc.edu",
    description="A CUDA-accelerated implementation of Conway's Game of Life",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sagniknath91/gameoflife-cuda",  # Link to your repo
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=["torch", "vispy"],  # Dependencies
    entry_points={
        "console_scripts": [
            "gameoflife=gameoflife_cuda.gameoflife_cuda:main",  # CLI entry point
        ],
    },
)
