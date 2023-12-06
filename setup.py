import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

setuptools.setup(
    name='medbench',
    version='0.1',
    author='Judith Abécassis',
    description='Benchmark mediation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages('.'),
    install_requires=[
        'pandas>=1.2.1',
        'scikit-learn>=0.22.1',
        'numpy>=1.19.2',
        'rpy2>=2.9.4',
        'scipy>=1.5.2',
        'seaborn>=0.11.1',
        'matplotlib>=3.3.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
