import setuptools
import warnings
import sys
import glob

required_packages =['numpy>=1.19.2', 'scipy>=1.4.0', 'tqdm>=4.41.1', 'torch>=1.9.0']

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="twoharmonics",
	version="0.1.0",
	author="Manvi Jain",
	author_email="manvijain368@gmail.com",
	maintainer = "Manvi Jain",
	maintainer_email = "manvijain368@gmail.com",
	description="Representation of precessing waveforms based on harmonic modes.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://git.ligo.org/manvij1612/twoharmonics",
	packages = setuptools.find_packages(),
	classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'pycbc',
        'notebook',
    ],
    package_data={
        '': ['notebooks/*.ipynb'],
    },
)
