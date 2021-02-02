# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optimize chains of single-qubit gates using Euler 1q decomposer"""

import logging

import numpy as np

from qiskit.quantum_info import Operator
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit.converters import circuit_to_dag

LOG = logging.getLogger(__name__)


class Optimize1qGatesDecomposition(TransformationPass):
    """Optimize chains of single-qubit gates by combining them into a single gate."""

    def __init__(self, basis=None):
        """Optimize1qGatesDecomposition initializer.

        Args:
            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects
                of this pass, the basis is the set intersection between the `basis` parameter
                and the Euler basis.
        """
        super().__init__()
        self.basis = None
        if basis:
            self.basis = []
            basis_set = set(basis)
            basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
            for basis_name, gates in basis_gates.items():
                if set(gates).issubset(basis_set):
                    for base in self.basis:
                        # check if the gates are a subset of another basis
                        if set(gates).issubset(set(basis_gates[base.basis])):
                            break
                        # check if gates are a superset of another basis
                        # and if so, remove that basis
                        elif set(basis_gates[base.basis]).issubset(set(gates)):
                            self.basis.remove(base)
                    # if not a subset, add it to the list
                    else:
                        self.basis.append(one_qubit_decompose.OneQubitEulerDecomposer(basis_name))

    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if not self.basis:
            LOG.info("Skipping pass because no basis is set")
            return dag
        runs = dag.collect_1q_runs()
        identity_matrix = np.eye(2)
        for run in runs:
            # Don't try to optimize a single 1q gate
            if len(run) <= 1:
                params = run[0].op.params
                # Remove single identity gates
                if len(params) > 0 and np.array_equal(run[0].op.to_matrix(),
                                                      identity_matrix):
                    dag.remove_op_node(run[0])
                continue

            new_circs = []
            operator = Operator(run[0].op)
            for gate in run[1:]:
                operator = operator.compose(gate.op)
            for decomposer in self.basis:
                new_circs.append(decomposer(operator))
            if new_circs:
                new_circ = min(new_circs, key=len)
                if len(run) > len(new_circ):
                    new_dag = circuit_to_dag(new_circ)
                    dag.substitute_node_with_dag(run[0], new_dag)
                    # Delete the other nodes in the run
                    for current_node in run[1:]:
                        dag.remove_op_node(current_node)
        return dag
