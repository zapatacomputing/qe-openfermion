import unittest
import random
import numpy as np
import os

import pyquil
from pyquil.paulis import sX, sY, sZ, sI

from openfermion import QubitOperator, IsingOperator

from openfermion.utils import qubit_operator_sparse

from ._io import (
    convert_qubitop_to_dict, save_qubit_operator, load_qubit_operator
)
from ._utils import (
    generate_random_qubitop, get_qubitop_from_coeffs_and_labels,
    evaluate_qubit_operator, get_qubitop_from_matrix, reverse_qubit_order,
    expectation, change_operator_type, evaluate_operator_for_parameter_grid
)


from zquantum.core.measurement import ExpectationValues
from zquantum.core.utils import RNDSEED, create_object
from zquantum.core.testing import create_random_qubitop, create_random_isingop
from zquantum.core.circuit import build_uniform_param_grid

class TestQubitOperator(unittest.TestCase):

    def test_build_qubitoperator_from_coeffs_and_labels(self):
        # Given
        test_op = QubitOperator(((0, 'Y'), (1, 'X'), (2, 'Z'), (4, 'X')), 3.j)
        coeffs = [3.j]
        labels = [[2, 1, 3, 0, 1]]

        # When
        build_op = get_qubitop_from_coeffs_and_labels(coeffs, labels)

        # Then
        self.assertEqual(test_op, build_op)

    def test_qubitop_matrix_converion(self):
        # Given
        m = 4
        n = 2**m
        TOL = 10**-15
        random.seed(RNDSEED)
        A = np.array([[random.uniform(-1,1) for x in range(n)] for y in range(n)])

        # When
        A_qubitop = get_qubitop_from_matrix(A)
        A_qubitop_matrix = np.array(qubit_operator_sparse(A_qubitop).todense())
        test_matrix = A_qubitop_matrix - A
        
        # Then
        for row in test_matrix:
            for elem in row:
                self.assertEqual(abs(elem)<TOL, True)

    def test_generate_random_qubitop(self):
        # Given
        nqubits = 4
        nterms = 5
        nlocality = 2
        max_coeff = 1.5
        fixed_coeff = False

        # When
        qubit_op = generate_random_qubitop(nqubits, nterms, nlocality, max_coeff, fixed_coeff)
        # Then
        self.assertEqual(len(qubit_op.terms), nterms)
        for term, coefficient in qubit_op.terms.items():
            for i in range(nlocality):
                self.assertLess(term[i][0], nqubits)
            self.assertEqual(len(term), nlocality)
            self.assertLessEqual(np.abs(coefficient), max_coeff)

        # Given
        fixed_coeff = True
        # When
        qubit_op = generate_random_qubitop(nqubits, nterms, nlocality, max_coeff, fixed_coeff)
        # Then
        self.assertEqual(len(qubit_op.terms), nterms)
        for term, coefficient in qubit_op.terms.items():
            self.assertEqual(np.abs(coefficient), max_coeff)

    def test_evaluate_qubit_operator(self):
        # Given
        qubit_op = QubitOperator('0.5 [] + 0.5 [Z1]')
        expectation_values = ExpectationValues([0.5, 0.5])
        # When
        value_estimate = evaluate_qubit_operator(qubit_op, expectation_values)
        # Then
        self.assertAlmostEqual(value_estimate.value, 0.5)

    def test_evaluate_operator_for_parameter_grid(self):
        # Given
        ansatz = {'ansatz_type': 'singlet UCCSD', 'ansatz_module': 'zquantum.qaoa.ansatz', 'ansatz_func': 'build_qaoa_circuit', 'ansatz_grad_func': 'build_qaoa_circuit_grads', 'supports_simple_shift_rule': False, 'ansatz_kwargs': {'hamiltonians': [{'schema': 'zapata-v1-qubit_op', 'terms': [{'pauli_ops': [], 'coefficient': {'real': 0.5}}, {'pauli_ops': [{'qubit': 1, 'op': 'Z'}], 'coefficient': {'real': 0.5}}]}, {'schema': 'zapata-v1-qubit_op', 'terms': [{'pauli_ops': [{'qubit': 0, 'op': 'X'}], 'coefficient': {'real': 1.0}}, {'pauli_ops': [{'qubit': 1, 'op': 'X'}], 'coefficient': {'real': 1.0}}]}]}, 'n_params': [2]}
        grid = build_uniform_param_grid(ansatz, 1, 0, np.pi, np.pi/10)
        backend = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumSimulator'})
        op = QubitOperator('0.5 [] + 0.5 [Z1]')
        previous_layer_parameters = [1, 1]
        # When
        parameter_grid_evaluation, optimal_parameters = evaluate_operator_for_parameter_grid(ansatz, grid, backend, op, previous_layer_params=previous_layer_parameters)
        # Then (for brevity, only check first and last evaluations)
        self.assertIsInstance(parameter_grid_evaluation[0]['value'].value, float)
        self.assertEqual(parameter_grid_evaluation[0]['parameter1'], 0)
        self.assertEqual(parameter_grid_evaluation[0]['parameter2'], 0)
        self.assertIsInstance(parameter_grid_evaluation[99]['value'].value, float)
        self.assertEqual(parameter_grid_evaluation[99]['parameter1'], np.pi-np.pi/10)
        self.assertEqual(parameter_grid_evaluation[99]['parameter2'], np.pi-np.pi/10)
        
        self.assertEqual(len(optimal_parameters), 4)
        self.assertEqual(optimal_parameters[0], 1)
        self.assertEqual(optimal_parameters[1], 1)

    def test_reverse_qubit_order(self):
        # Given
        op1 = QubitOperator('[Z0 Z1]')
        op2 = QubitOperator('[Z1 Z0]')

        # When/Then
        self.assertEqual(op1, reverse_qubit_order(op2))

        # Given
        op1 = QubitOperator('Z0')
        op2 = QubitOperator('Z1')

        # When/Then
        self.assertEqual(op1, reverse_qubit_order(op2, n_qubits=2))
        self.assertEqual(op2, reverse_qubit_order(op1, n_qubits=2))

    def test_expectation(self):
        """Check <Z0> and <Z1> for the state |100>"""
        # Given
        wf = pyquil.wavefunction.Wavefunction([0, 1, 0, 0, 0, 0, 0, 0])
        op1 = QubitOperator('Z0')
        op2 = QubitOperator('Z1')
        # When 
        exp_op1 = expectation(op1, wf)
        exp_op2 = expectation(op2, wf)
        
        # Then
        self.assertAlmostEqual(-1, exp_op1)
        self.assertAlmostEqual(1, exp_op2)

    def test_change_operator_type(self):
        # Given
        operator1 = QubitOperator('Z0 Z1', 4.5)
        operator2 = IsingOperator('Z0 Z1', 4.5)
        operator3 = IsingOperator()
        operator4 = IsingOperator('Z0', 0.5) + IsingOperator('Z1', 2.5)
        # When 
        new_operator1 = change_operator_type(operator1, IsingOperator)
        new_operator2 = change_operator_type(operator2, QubitOperator)
        new_operator3 = change_operator_type(operator3, QubitOperator)
        new_operator4 = change_operator_type(operator4, QubitOperator)
        
        # Then
        self.assertEqual(IsingOperator('Z0 Z1', 4.5), new_operator1)
        self.assertEqual(QubitOperator('Z0 Z1', 4.5), new_operator2)
        self.assertEqual(QubitOperator(), new_operator3)
        self.assertEqual(QubitOperator('Z0', 0.5) + QubitOperator('Z1', 2.5), new_operator4)
