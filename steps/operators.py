from qeopenfermion import (
    get_fermion_number_operator as _get_fermion_number_operator,
    get_diagonal_component as _get_diagonal_component,
    save_interaction_operator,
    load_interaction_operator,
    load_qubit_operator,
    save_qubit_operator,
)


def get_fermion_number_operator(number_of_qubits, number_of_particles="None"):
    if number_of_particles == "None":
        number_of_particles = None
    else:
        number_of_particles = int(number_of_particles)

    number_op = _get_fermion_number_operator(number_of_qubits, number_of_particles)
    save_interaction_operator(number_op, "number-operator.json")


def get_diagonal_component(interaction_operator):
    interaction_operator = load_interaction_operator(interaction_operator)
    diagonal_operator, remainder_operator = _get_diagonal_component(interaction_operator)
    save_interaction_operator(diagonal_operator, "diagonal-operator.json")
    save_interaction_operator(remainder_operator, "remainder-operator.json")


def interpolate_qubit_operators(
    reference_qubit_operator, target_qubit_operator, epsilon=0.5
):
    reference_qubit_operator = load_qubit_operator(reference_qubit_operator)
    target_qubit_operator = load_qubit_operator(target_qubit_operator)

    if epsilon > 1.0 or epsilon < 0.0:
        raise ValueError("epsilon must be in the range [0.0, 1.0]")

    output_qubit_operator = (
        epsilon * target_qubit_operator + (1.0 - epsilon) * reference_qubit_operator
    )

    save_qubit_operator(output_qubit_operator, "qubit-operator.json")
