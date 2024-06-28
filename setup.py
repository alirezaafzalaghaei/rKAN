from setuptools import setup

setup(
    name='rkan',
    version='0.0.3',
    packages=["rkan"],
    description="Rational Kolmogorov-Arnold Network (rKAN)",
    author='Alireza Afzal Aghaei',
    author_email='alirezaafzalaghaei@gmail.com',
    url='https://github.com/alirezaafzalaghaei/rkan',
    python_needed='>=3.9',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD License",
    ],
    license='BSD',
)
