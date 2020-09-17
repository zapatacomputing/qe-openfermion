from zquantum.core.utils import save_timing
from qeopenfermion import load_interaction_operator, save_qubit_operator
from openfermion import jordan_wigner, bravyi_kitaev, get_fermion_operator
import time
import os


def transform_interaction_operator(transformation, input_operator):
    input_operator = load_interaction_operator(input_operator)

    if transformation == "Jordan-Wigner":
        transformation = jordan_wigner
    elif transformation == "Bravyi-Kitaev":
        input_op = get_fermion_operator(input_op)
        transformation = bravyi_kitaev
    else:
        raise RuntimeError("Unrecognized transformation ", transformation)

    start_time = time.time()
    transformed_operator = transformation(input_operator)
    walltime = time.time() - start_time

    save_qubit_operator(transformed_operator, "transformed-operator.json")
    save_timing(walltime, "timing.json")
