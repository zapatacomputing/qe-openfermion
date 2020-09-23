# qe-openfermion

## What is it?

`qe-openfermion` is an [Orquestra](https://www.orquestra.io) resource that allows workflows to use the [OpenFermion](https://github.com/quantumlib/OpenFermion) library.

[Orquestra](https://www.orquestra.io) is a platform for performing computations on quantum computers developed by [Zapata Computing](https://www.zapatacomputing.com).

## Usage

### Workflow
In order to use `qe-openfermion` in your workflow, you need to add it as a `resource` in your Orquestra workflow:

```yaml
imports:
- name: qe-openfermion
  type: git
  parameters:
    url: "git@github.com:zapatacomputing/qe-openfermion.git"
    branch: "master"
```

and then include it in the `imports` argument of your `step`:

```yaml
- name: create-molecule
  config:
    runtime:
      language: python3
      imports: [qe-openfermion]
```

Once that is done you can:
- execute any function from the `steps/` directory as a `step`
- have the Python code in your step import the `qeopenfermion` module (see below)

### Python

Here's an example how to do use methods from `qe-openfermion` in a task:

```python
from qeopenfermion import load_qubit_operator
operator = load_qubit_operator('operator.json')
```

Even though it's intended to be used with Orquestra, `qe-openfermion` can be used as a standalone Python module.
This can be done by running `pip install .` from the `src/` directory.

## Development and Contribution

- If you'd like to report a bug/issue please create a new issue in this repository.
- If you'd like to contribute, please create a pull request.

### Running tests

Unit tests for this project can be run using `pytest .` from the main directory.

