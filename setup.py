import os
from setuptools import setup, find_packages

# Get CUDA version from the env variable, defaulting to '11.6' if not set
cuda_version = os.getenv('CUDA_VERSION', '11.6')

# Strip the period from the version string (e.g., '11.6' -> '116')
cuda_version = cuda_version.replace('.', '')

# Format version to get prebuilt wheel
cupy_package = f'cupy-cuda{cuda_version}'

setup(
    name='oct_tissuemasking',
    version='0.0.7',
    description='A PyTorch based package for automated OCT tissue masking.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Etienne Chollet',
    author_email='etiennepchollet@gmail.com',
    url='https://github.com/EtienneChollet/oct_tissuemasking',
    packages=find_packages(),
    entry_points={
            'console_scripts': [
                'oct_tissuemasking=oct_tissuemasking:app'
            ]
        },
    install_requires=[
        cupy_package,
        'torch',
        'torchvision',
        'torchaudio',
        'torchmetrics',
        'jitfields',
        'torch-interpol',
        'torch-distmap',
        'nibabel',
        'pytorch-lightning',
        'scikit-learn',
        'matplotlib',
        'tensorboard',
        'pandas',
        'cyclopts'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='~=3.9',
)
