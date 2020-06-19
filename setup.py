from setuptools import setup, find_packages
#from distutils.core import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='rectorch',
    packages=find_packages(exclude=['build', 'doc', 'templates']),
    version='0.0.9b',
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "Bottleneck",
        "munch"
    ],
    python_requires='>=3.6',
    license="MIT",
    description='rectorch: state-of-the-art recsys approaches implemented in pytorch.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Mirko Polato',
    author_email='mak1788@gmail.com',
    url='https://github.com/makgyver/rectorch',
    download_url='https://github.com/makgyver/rectorch',
    keywords=['recommender-system', 'pytorch', 'machine-learning', 'algorithm', 'variational-autoencoder', 'gan'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
    ]
)
