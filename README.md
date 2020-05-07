# qe-openfermion

## What is it?

`qe-openfermion` is an [Orquestra](https://www.zapatacomputing.com/orquestra/) resource that allows workflows to use the [OpenFermion](https://github.com/quantumlib/OpenFermion) library.

[Orquestra](https://www.zapatacomputing.com/orquestra/) is a platform for performing computations on quantum computers developed by [Zapata Computing](https://www.zapatacomputing.com).

## Usage

### Workflow
In order to use `qe-openfermion` in your workflow, you need to add it as a `resource` in your Orquestra workflow:

```yaml
resources:
- name: qe-openfermion
  type: git
  parameters:
    url: "git@github.com:zapatacomputing/qe-openfermion.git"
    branch: "master"
```

and then import in the `resources` argument of your `task`:

```yaml
- - name: my-task
    template: template-1
    arguments:
      parameters:
      - param_1: 1
      - resources: [qe-openfermion]
```

Once that is done you can:
- use any template from the `templates/` directory
- use tasks which import `qeopenfermion` in the python code (see below)

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

