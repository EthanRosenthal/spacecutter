import os
from setuptools import setup, find_packages

from spacecutter import __version__


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

with open(os.path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(os.path.join(here, 'test-requirements.txt')) as f:
    test_requirements = f.read().splitlines()


setup(
    name='spacecutter',
    version=__version__,
    description='Ordinal regression models in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/EthanRosenthal/spacecutter',
    author='Ethan Rosenthal',
    author_email='ethanrosenthal@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='machine learning statistics',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    extras_require={
        'test': test_requirements,
    },
)
