import setuptools


with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="jnumpy",
    version="1.0.0",
    author="Jacob Valdez",
    author_email="jacobfv@msn.com",
    description="Jacob's numpy library for machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JacobFV/jnumpy",
    packages=["jnumpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
    ],
    keywords="numpy machine learning automatic differentiation",
    project_urls={
        'Homepage': "https://github.com/JacobFV/jnumpy",
    },
)