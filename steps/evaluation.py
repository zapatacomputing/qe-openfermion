from zquantum.core.measurement import save_expectation_values, save_wavefunction
from zquantum.core.circuit import (
    load_circuit,
    load_parameter_grid,
    load_circuit_template_params,
    save_circuit_template_params,
)
from zquantum.core.utils import create_object, ValueEstimate, save_value_estimate
from qeopenfermion import (
    load_qubit_operator,
    evaluate_operator_for_parameter_grid as _evaluate_operator_for_parameter_grid,
    save_parameter_grid_evaluation,
)
from openfermion.utils import (
    qubit_operator_sparse,
    jw_get_ground_state_at_particle_number as _jw_get_ground_state_at_particle_number,
)
from pyquil.wavefunction import Wavefunction
import json
import os


def get_expectation_values_for_qubit_operator(backend_specs, circuit, qubit_operator):
    circuit = load_circuit(circuit)
    qubit_operator = load_qubit_operator(qubit_operator)
    backend = create_object(json.loads(backend_specs))
    expectation_values = backend.get_expectation_values(circuit, qubit_operator)
    save_expectation_values(expectation_values, "expectation-values.json")


def evaluate_operator_for_parameter_grid(
    ansatz_specs,
    backend_specs,
    grid,
    operator,
    fixed_parameters="None",
):

    ansatz = create_object(json.loads(ansatz_specs))
    backend = create_object(json.loads(backend_specs))

    grid = load_parameter_grid(grid)
    operator = load_qubit_operator(operator)

    if fixed_parameters != "None":
        if type(fixed_parameters) == str:
            if os.path.exists(fixed_parameters):
                fixed_parameters = load_circuit_template_params(fixed_parameters)
    else:
        fixed_parameters = []

    (
        parameter_grid_evaluation,
        optimal_parameters,
    ) = _evaluate_operator_for_parameter_grid(
        ansatz, grid, backend, operator, previous_layer_params=fixed_parameters
    )

    save_parameter_grid_evaluation(
        parameter_grid_evaluation, "parameter-grid-evaluation.json"
    )
    save_circuit_template_params(optimal_parameters, "/app/optimal-parameters.json")


def jw_get_ground_state_at_particle_number(particle_number, qubit_operator):
    qubit_operator = load_qubit_operator(qubit_operator)
    sparse_matrix = qubit_operator_sparse(qubit_operator)

    ground_energy, ground_state_amplitudes = _jw_get_ground_state_at_particle_number(
        sparse_matrix, particle_number
    )
    ground_state = Wavefunction(ground_state_amplitudes)
    value_estimate = ValueEstimate(ground_energy)

    save_wavefunction(ground_state, "ground-state.json")
    save_value_estimate(value_estimate, "value-estimate.json")
