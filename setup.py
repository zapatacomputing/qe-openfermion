import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="qe-openfermion",
    version="0.1.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Integrations for deploying openfermion on Orquestra Quantum Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/qe-openfermion",
    packages=["qeopenfermion"],
    package_dir={"": "src/python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "pytest>=5.3.5",
        "numpy>=1.18.1",
        "openfermion>=1.0.0",
        "python-rapidjson",
        "pyquil>=2.17.0",
        "z-quantum-core",
    ],
)
