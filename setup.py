from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reticular-sae",
    version="0.1",
    author="John J. Yang",
    author_email="john@reticular.ai",  # Add if available
    description="A toolkit for extracting, analyzing, and visualizing interpretable features from protein language models using sparse autoencoders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnyang101/reticular-sae",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)