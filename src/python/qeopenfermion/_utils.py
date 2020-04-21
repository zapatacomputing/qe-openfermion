from openfermion import QubitOperator, count_qubits
from openfermion.utils import expectation as openfermion_expectation
from openfermion.transforms import get_sparse_operator
from pyquil.paulis import PauliSum, PauliTerm
import numpy as np
import random
import copy
from typing import List, Union, Optional

from zquantum.core.circuit import build_ansatz_circuit
from zquantum.core.utils import bin2dec, dec2bin, ValueEstimate
from zquantum.core.measurement import ExpectationValues
from openfermion import count_qubits

def get_qubitop_from_matrix(operator: List[List]) -> QubitOperator:
	r"""Expands a 2^n by 2^n matrix into n-qubit Pauli basis. The runtime of
	this function is O(2^2n).

	Args:
		operator: a list of lists (rows) representing a 2^n by 2^n
			matrix.

	Returns:
		A QubitOperator instance corresponding to the expansion of
		the input operator as a sum of Pauli strings:

		O = 2^-n \sum_P tr(O*P) P
	"""

	nrows = len(operator)
	ncols = len(operator[0])

	# Check if the input operator is square
	if nrows != ncols:
		raise Exception('The input operator is not square')

	# Check if the dimensions are powers of 2
	if not ( ((nrows & (nrows-1)) == 0) and nrows > 0):
		raise Exception('The number of rows is not a power of 2')
	if not ( ((ncols & (ncols-1)) == 0) and ncols > 0):
		raise Exception('The number of cols is not a power of 2')

	n = int(np.log2(nrows))	# number of qubits

	def decode(bit_string): # Helper function for converting any 2n-bit
				# string to a label vector representing a Pauli
				# string of length n

		if len(bit_string) != 2*n:
			raise Exception('LH_expand:decode: input bit string length not 2n')

		output_label = list(np.zeros(n))
		for i in range(0, n):
			output_label[i] = bin2dec(bit_string[2*i:2*i+2])

		return output_label

	def trace_product(label_vec): # Helper function for computing tr(OP)
				# where O is the input operator and P is a
				# Pauli string operator

		def f(j): # Function which computes the index of the nonzero
				# element in P for a given column j

			j_str = dec2bin(j, n)
			for index in range(0, n):
				if label_vec[index] in [1,2]: # flip if X or Y
					j_str[index] = int(not j_str[index])
			return bin2dec(j_str)

		def nz(j): # Function which computes the value of the nonzero
				# element in P on the column j

			val_nz = 1.0
			j_str = dec2bin(j, n)
			for index in range(0, n):
				if label_vec[index] == 2:
					if j_str[index] == 0:
						val_nz = val_nz * (1j)
					if j_str[index] == 1:
						val_nz = val_nz * (-1j)
				if label_vec[index] == 3:
					if j_str[index] == 1:
						val_nz = val_nz * (-1)
			return val_nz

		# Compute the trace
		tr = 0.0
		for j in range(0, 2**n): # loop over the columns
			tr = tr + operator[j][f(j)] * nz(j)

		return tr / 2**n

	# Expand the operator in Pauli basis
	coeffs = list(np.zeros(4**n))
	labels = list(np.zeros(4**n))
	for i in range(0, 4**n): # loop over all 2n-bit strings
		current_string = dec2bin(i, 2*n) # see util.py
		current_label = decode(current_string)
		coeffs[i] = trace_product(current_label)
		labels[i] = current_label

	return get_qubitop_from_coeffs_and_labels(coeffs, labels)


def get_qubitop_from_coeffs_and_labels(coeffs: List[float], labels: List[List[int]]) -> QubitOperator:
	"""Generates a QubitOperator based on a coefficient vector and
	a label matrix.

	Args:
		coeffs: a list of floats representing the coefficients
			for the terms in the Hamiltonian
		labels: a list of lists (a matrix) where each list
			is a vector of integers representing the Pauli
			string. See pauliutil.py for details.

	Example:

		The Hamiltonian H = 0.1 X1 X2 - 0.4 Y1 Y2 Z3 Z4 can be
		initiated by calling

		H = QubitOperator([0.1, -0.4],\    # coefficients
					[[1 1 0 0],\  # label matrix
						[2 2 3 3]])
	"""

	output = QubitOperator()
	for i in range(0, len(labels)):
		string_term = ''
		for ind, elem in enumerate(labels[i]):
			pauli_symbol = ''
			if elem == 1:
				pauli_symbol = 'X' + str(ind) + ' '
			if elem == 2:
				pauli_symbol = 'Y' + str(ind) + ' '
			if elem == 3:
				pauli_symbol = 'Z' + str(ind) + ' '
			string_term += pauli_symbol

		output += coeffs[i] * QubitOperator(string_term)

	return output


def generate_random_qubitop(nqubits: int, 
							nterms: int, 
							nlocality: int, 
							max_coeff: float, 
							fixed_coeff: bool = False) -> QubitOperator:
	"""Generates a Hamiltonian with term coefficients uniformly distributed
	in [-max_coeff, max_coeff].

	Args:
		nqubits - number of qubits
		nterms	- number of terms in the Hamiltonian
		nlocality - locality of the Hamiltonian
		max_coeff - bound for generating the term coefficients
		fixed_coeff (bool) - If true, all the terms are assign the
			max_coeff as coefficient.

	Returns:
		A QubitOperator with the appropriate coefficient vector
		and label matrix.
	"""
	# generate random coefficient vector
	if fixed_coeff:
		coeffs = [max_coeff] * nterms
	else:
		coeffs = list(np.zeros(nterms))
		for j in range(0, nterms):
			coeffs[j] = random.uniform(-max_coeff, max_coeff)

	# generate random label vector
	labels = list(np.zeros(nterms, dtype=int))
	label_set = set()
	j = 0
	while j < nterms:
		inds_nontrivial = sorted(random.sample(range(0, nqubits),\
			nlocality))
		label = list(np.zeros(nqubits, dtype=int))
		for ind in inds_nontrivial:
			label[ind] = random.randint(1, 3)
		if str(label) not in label_set:
			labels[j] = label
			j += 1
			label_set.add(str(label))
	return get_qubitop_from_coeffs_and_labels(coeffs, labels)


def evaluate_qubit_operator(qubit_operator: QubitOperator, 
							expectation_values: ExpectationValues) -> ValueEstimate:
	"""Evaluate the expectation value of a qubit operator using
	expectation values for the terms.

	Args:
		qubit_operator (openfermion.QubitOperator): the operator
		expectation_values (core.measurement.ExpectationValues): the expectation values

	Returns:
		value_estimate (zquantum.core.utils.ValueEstimate): stores the value of the expectation and its
			 precision
	"""

	# Sum the contributions from all terms
	total = 0

	# Add all non-trivial terms
	term_index = 0
	for term in qubit_operator.terms:
		total += np.real(qubit_operator.terms[term]*expectation_values.values[term_index])
		term_index += 1

	value_estimate = ValueEstimate(total)
	return value_estimate


def evaluate_operator_for_parameter_grid(ansatz, grid, backend, operator,
    previous_layer_params=[]):
	"""Evaluate the expectation value of an operator for every set of circuit
	parameters in the parameter grid.

	Args:
		ansatz (dict): the ansatz
		grid (zquantum.core.circuit.ParameterGrid): The parameter grid containing
			the parameters for the last layer of the ansatz
        backend (zquantum.core.interfaces.backend.QuantumSimulator): the backend 
			to run the circuits on 
		operator (openfermion.ops.QubitOperator): the operator
		previous_layer_params (array): A list of the parameters for previous layers
			of the ansatz

	Returns:
		value_estimate (zquantum.core.utils.ValueEstimate): stores the value of the expectation and its
			 precision
	"""
	parameter_grid_evaluation = []
	for last_layer_params in grid.params_list:
        # Build the ansatz circuit
		params = np.concatenate((np.asarray(previous_layer_params), np.asarray(last_layer_params)))

        # Build the ansatz circuit
		circuit = build_ansatz_circuit(ansatz, params)

		expectation_values = backend.get_expectation_values(circuit, operator)
		value_estimate = ValueEstimate(sum(expectation_values.values))
		parameter_grid_evaluation.append({'value': value_estimate, 'parameter1': last_layer_params[0], 'parameter2': last_layer_params[1]})
		
	return parameter_grid_evaluation


def reverse_qubit_order(qubit_operator:QubitOperator, n_qubits:Optional[int]=None):
    """Reverse the order of qubit indices in a qubit operator.

    Args:
        qubit_operator (openfermion.QubitOperator): the operator
        n_qubits (int): total number of qubits. Needs to be provided when 
					the size of the system of interest is greater than the size of qubit operator (optional)

    Returns:
        reversed_op (openfermion.ops.QubitOperator)
    """

    reversed_op = QubitOperator()

    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    if n_qubits < count_qubits(qubit_operator):
        raise ValueError('Invalid number of qubits specified.')

    for term in qubit_operator.terms:
        new_term = []
        for factor in term:
            new_factor = list(factor)
            new_factor[0] = n_qubits - 1 - new_factor[0]
            new_term.append(tuple(new_factor))
        reversed_op += QubitOperator(tuple(new_term), qubit_operator.terms[term])
    return reversed_op


def expectation(qubit_op, wavefunction, reverse_operator=True):
	"""Get the expectation value of a qubit operator with respect to a wavefunction.
	Args:
		qubit_op (openfermion.ops.QubitOperator): the operator
		wavefunction (pyquil.wavefunction.Wavefunction): the wavefunction
		reverse_operator (boolean): whether to reverse order of qubit operator
			before computing expectation value. This should be True if the convention
			of the basis states used for the wavefunction is the opposite of the one in
			the qubit operator. This is the case, e.g. when the wavefunction comes from
			Pyquil.
	Returns:
		complex: the expectation value
	"""
	n_qubits = wavefunction.amplitudes.shape[0].bit_length() - 1
	
	# Convert the qubit operator to a sparse matrix. Note that the qubit indices
	# must be reversed because OpenFermion and pyquil use different conventions
	# for how to order the computational basis states!
	if reverse_operator:
		qubit_op = reverse_qubit_order(qubit_op, n_qubits=n_qubits)
	sparse_op = get_sparse_operator(qubit_op, n_qubits=n_qubits)
	
	# Computer the expectation value
	exp_val = openfermion_expectation(sparse_op, wavefunction.amplitudes)
	return exp_val
	

def change_operator_type(operator, operatorType):
	'''Take an operator and attempt to cast it to an operator of a different type

	Args:
		operator: The operator
		operatorType: The type of the operator that the original operator is
			cast to
	Returns:
		An operator with type operatorType
	'''
	new_operator = operatorType()
	for op in operator.terms:
		new_operator += operatorType(tuple(op), operator.terms[op])
	
	return new_operator
