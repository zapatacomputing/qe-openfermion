import setuptools
import os

readme_path = os.path.join("..", "README.md")
with open(readme_path, "r") as f:
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
    packages=['qeopenfermion'],
    package_dir={'' : 'python'},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'pytest>=5.3.5',
        'numpy>=1.18.1',
        'scipy<1.3.0', # openfermion 0.10.0 is incompatible with scipy 1.3 because of comb function
        'openfermion>=0.10.0',
        'python-rapidjson',
        'z-quantum-core'
    ]
)