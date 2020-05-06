import unittest
import subprocess
import os
import numpy as np
from openfermion import (
    QubitOperator, InteractionOperator, FermionOperator, IsingOperator,
    get_interaction_operator, hermitian_conjugated
)
from zquantum.core.circuit import build_uniform_param_grid, save_circuit_template_params
from zquantum.core.utils import create_object
from ._utils import evaluate_operator_for_parameter_grid
from ._io import (
    load_qubit_operator, save_qubit_operator, load_interaction_operator,
    save_interaction_operator, convert_qubitop_to_dict, convert_dict_to_qubitop, 
    convert_interaction_op_to_dict, convert_dict_to_interaction_op,
    convert_isingop_to_dict, convert_dict_to_isingop, 
    save_ising_operator, load_ising_operator, save_parameter_grid_evaluation
)


class TestQubitOperator(unittest.TestCase):

    def test_qubitop_to_dict_io(self):
        # Given
        qubit_op = QubitOperator(((0, 'Y'), (1, 'X'), (2, 'Z'), (4, 'X')), 3.j)
        qubit_op += hermitian_conjugated(qubit_op)
        
        # When
        qubitop_dict = convert_qubitop_to_dict(qubit_op)
        recreated_qubit_op = convert_dict_to_qubitop(qubitop_dict)

        # Then
        self.assertEqual(recreated_qubit_op, qubit_op)

    def test_qubit_operator_io(self):
        # Given
        qubit_op = QubitOperator(((0, 'Y'), (3, 'X'), (8, 'Z'), (11, 'X')), 3.j)

        # When
        save_qubit_operator(qubit_op, 'qubit_op.json')
        loaded_op = load_qubit_operator('qubit_op.json')

        # Then
        self.assertEqual(qubit_op, loaded_op)
        os.remove('qubit_op.json')

    def test_interaction_op_to_dict_io(self):
        # Given
        test_op = FermionOperator('1^ 2^ 3 4')
        test_op += hermitian_conjugated(test_op)
        interaction_op = get_interaction_operator(test_op)
        interaction_op.constant = 0.0

        # When
        interaction_op_dict = convert_interaction_op_to_dict(interaction_op)
        recreated_interaction_op = convert_dict_to_interaction_op(interaction_op_dict)

        # Then
        self.assertEqual(recreated_interaction_op, interaction_op)

    def test_interaction_operator_io(self):
        # Given
        test_op = FermionOperator('1^ 2^ 3 4')
        test_op += hermitian_conjugated(test_op)
        interaction_op = get_interaction_operator(test_op)
        interaction_op.constant = 0.0
        
        # When
        save_interaction_operator(interaction_op, 'interaction_op.json')
        loaded_op = load_interaction_operator('interaction_op.json')

        # Then
        self.assertEqual(interaction_op, loaded_op)
        os.remove('interaction_op.json')

    def test_qubitop_io(self):
        # Given
        qubit_op = QubitOperator(((0, 'Y'), (1, 'X'), (2, 'Z'), (4, 'X')), 3.j)
        
        # When
        save_qubit_operator(qubit_op, 'qubit_op.json')
        loaded_op = load_qubit_operator('qubit_op.json')

        # Then
        self.assertEqual(qubit_op, loaded_op)
        os.remove('qubit_op.json')

    def test_isingop_to_dict_io(self):
        # Given
        ising_op = IsingOperator('[] + 3[Z0 Z1] + [Z1 Z2]')
        
        # When
        isingop_dict = convert_isingop_to_dict(ising_op)
        recreated_isingop = convert_dict_to_isingop(isingop_dict)

        # Then
        self.assertEqual(recreated_isingop, ising_op)

    def test_isingop_io(self):
        # Given
        ising_op = IsingOperator('[] + 3[Z0 Z1] + [Z1 Z2]')
        
        # When
        save_ising_operator(ising_op, 'ising_op.json')
        loaded_op = load_ising_operator('ising_op.json')

        # Then
        self.assertEqual(ising_op, loaded_op)
        os.remove('ising_op.json')
        

    def test_save_parameter_grid_evaluation(self):
        # Given
        ansatz = {'ansatz_type': 'singlet UCCSD', 'ansatz_module': 'zquantum.qaoa.ansatz', 'ansatz_func': 'build_qaoa_circuit', 'ansatz_grad_func': 'build_qaoa_circuit_grads', 'supports_simple_shift_rule': False, 'ansatz_kwargs': {'hamiltonians': [{'schema': 'zapata-v1-qubit_op', 'terms': [{'pauli_ops': [], 'coefficient': {'real': 0.5}}, {'pauli_ops': [{'qubit': 1, 'op': 'Z'}], 'coefficient': {'real': 0.5}}]}, {'schema': 'zapata-v1-qubit_op', 'terms': [{'pauli_ops': [{'qubit': 0, 'op': 'X'}], 'coefficient': {'real': 1.0}}, {'pauli_ops': [{'qubit': 1, 'op': 'X'}], 'coefficient': {'real': 1.0}}]}]}, 'n_params': [2]}
        grid = build_uniform_param_grid(ansatz, 1, 0, np.pi, np.pi/10)
        backend = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumSimulator'})
        op = QubitOperator('0.5 [] + 0.5 [Z1]')
        parameter_grid_evaluation, optimal_parameters = evaluate_operator_for_parameter_grid(ansatz, grid, backend, op)
        # When
        save_parameter_grid_evaluation(parameter_grid_evaluation, "parameter-grid-evaluation.json")
        save_circuit_template_params(optimal_parameters, "optimal-parameters.json")
        # Then 
        # TODO

    def tearDown(self):
        subprocess.run(["rm", "parameter-grid-evaluation.json", "optimal-parameters.json"])
