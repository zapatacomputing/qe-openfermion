import unittest
import random
import numpy as np
import os

import pyquil
from pyquil.paulis import sX, sY, sZ, sI
from openfermion import QubitOperator
import openfermion
from ._io import (
    convert_qubitop_to_dict, save_qubit_operator, load_qubit_operator
)
from ._utils import (
    generate_random_qubitop, get_qubitop_from_coeffs_and_labels,
    evaluate_qubit_operator, get_qubitop_from_matrix
)


from zquantum.core.measurement import ExpectationValues
from zquantum.core.utils import RNDSEED
from zquantum.core.testing import create_random_qubitop, create_random_isingop

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
        A_qubitop_matrix = np.array(openfermion.utils.qubit_operator_sparse(A_qubitop).todense())
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
        qubit_op = openfermion.QubitOperator('0.5 [] + 0.5 [Z1]')
        expectation_values = ExpectationValues([0.5, 0.5])
        # When
        value_estimate = evaluate_qubit_operator(qubit_op, expectation_values)
        # Then
        self.assertAlmostEqual(value_estimate.value, 0.5)
