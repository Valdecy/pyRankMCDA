from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pyRankMCDA',
    version='2.1.4',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyRankMCDA',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'adjustText',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'seaborn',
        'scipy',
        'scikit-learn'
    ],
    zip_safe=True,
    description='A Rank Aggegation Library for MCDA problems',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
