from setuptools import setup, find_packages
#from distutils.core import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='rectorch',
    packages=find_packages(exclude=['build', 'doc', 'templates', 'tests']),
    include_package_data=True,
    version='0.9.0dev',
    install_requires=[
        "pandas>=1.1.0",
        "numpy>=1.17.2",
        "scipy>=1.4.1",
        "torch>=1.2.0",
        "Bottleneck>=1.2.1",
        "munch>=2.5.0",
        "cvxopt>=1.2.3",
        "hyperopt>=0.1.2"
    ],
    python_requires='>=3.5',
    license="MIT",
    description='rectorch: state-of-the-art recsys approaches implemented in pytorch.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Mirko Polato',
    author_email='mak1788@gmail.com',
    url='https://github.com/makgyver/rectorch',
    download_url='https://github.com/makgyver/rectorch',
    keywords=['recommender-system', 'pytorch', 'machine-learning', 'algorithm', 'variational-autoencoder', 'gan', 'collaborative-filtering', 'top-n recommendation'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
    ]
)
