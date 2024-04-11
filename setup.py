from glob import glob
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyterrier_cohere",
    version="0.0.1",
    author="Sean MacAvaney",
    author_email="sean.macavaney@glasgow.ac.uk",
    description="PyTerrier integration with Cohere",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanmacavaney/pyterrier_cohere",
    packages=setuptools.find_packages(include=['pyterrier_cohere']),
    install_requires=list(open('requirements.txt')),
    classifiers=[],
    python_requires='>=3.9',
    package_data={
        '': ['requirements.txt'],
    },
)
