# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=redefined-builtin

"""Object to represent the information at a node in the DAGCircuit."""

import warnings

from qiskit.exceptions import QiskitError
from qiskit.circuit import Instruction, Gate


class DAGNode:
    """Object to represent the information at a node in the DAGCircuit.

    It is used as the return value from `*_nodes()` functions and can
    be supplied to functions that take a node.
    """

    __slots__ = ["type", "_qargs", "cargs", "sort_key", "_node_id"]

    def __init__(self, qargs=None, cargs=None, nid=-1):
        """Create a node"""
        self._qargs = qargs if qargs is not None else []
        self.cargs = cargs if cargs is not None else []
        self.sort_key = str(self._qargs)
        self._node_id = nid

    @property
    def qargs(self):
        """
        Returns list of Qubit, else an empty list.
        """
        return self._qargs

    @qargs.setter
    def qargs(self, new_qargs):
        """Sets the qargs to be the given list of qargs."""
        self._qargs = new_qargs
        self.sort_key = str(new_qargs)

    def __lt__(self, other):
        return self._node_id < other._node_id

    def __gt__(self, other):
        return self._node_id > other._node_id

    def __str__(self):
        # TODO is this used anywhere other than in DAG drawing?
        # needs to be unique as it is what pydot uses to distinguish nodes
        return str(id(self))

    @staticmethod
    def semantic_eq(node1, node2, bit_indices1=None, bit_indices2=None):
        """
        Check if DAG nodes are considered equivalent, e.g., as a node_match for nx.is_isomorphic.

        Args:
            node1 (DAGNode): A node to compare.
            node2 (DAGNode): The other node to compare.
            bit_indices1 (dict): Dictionary mapping Bit instances to their index
                within the circuit containing node1
            bit_indices2 (dict): Dictionary mapping Bit instances to their index
                within the circuit containing node2

        Return:
            Bool: If node1 == node2
        """
        if bit_indices1 is None or bit_indices2 is None:
            warnings.warn(
                "DAGNode.semantic_eq now expects two bit-to-circuit index "
                "mappings as arguments. To ease the transition, these will be "
                "pre-populated based on the values found in Bit.index and "
                "Bit.register. However, this behavior is deprecated and a future "
                "release will require the mappings to be provided as arguments.",
                DeprecationWarning,
            )

            bit_indices1 = {arg: arg for arg in node1.qargs + node1.cargs}
            bit_indices2 = {arg: arg for arg in node2.qargs + node2.cargs}

        node1_qargs = [bit_indices1[qarg] for qarg in node1.qargs]
        node1_cargs = [bit_indices1[carg] for carg in node1.cargs]

        node2_qargs = [bit_indices2[qarg] for qarg in node2.qargs]
        node2_cargs = [bit_indices2[carg] for carg in node2.cargs]

        if isinstance(node1, OpNode) and isinstance(node2, OpNode):
            # For barriers, qarg order is not significant so compare as sets
            if "barrier" == node1.name == node2.name:
                return set(node1_qargs) == set(node2_qargs)

            if type(node1) == type(node2):
                if node1.name == node2.name:
                    if node1_qargs == node2_qargs:
                        if node1_cargs == node2_cargs:
                            if node1.condition == node2.condition:
                                return True
        elif ((isinstance(node1, InNode) and isinstance(node2, InNode))
              or (isinstance(node1, OutNode) and isinstance(node2, OutNode))):
            if node1_qargs == node2_qargs:
                if node1_cargs == node2_cargs:
                    if bit_indices1.get(node1.wire, None) == bit_indices2.get(
                        node2.wire, None
                    ):
                        return True
        else:
            return False



class OpNode(DAGNode, Gate, Instruction):
    """Object to represent the information at a node in the DAGCircuit.

    It is used as the return value from `*_nodes()` functions and can
    be supplied to functions that take a node.
    """

    def __init__(self, op, qargs=None, cargs=None):
        """Create a node"""
        DAGNode.__init__(self, qargs, cargs)
        if isinstance(op, Gate):
            Gate.__init__(self, op.name, num_qubits=len(qargs), params=op.params)
        else:
            Instruction.__init__(self, op.name, num_qubits=len(qargs), num_clbits=len(cargs), params=op.params)

class InNode(DAGNode):
    """Object to represent the information at a node in the DAGCircuit.

    It is used as the return value from `*_nodes()` functions and can
    be supplied to functions that take a node.
    """

    __slots__ = ["wire"]

    def __init__(self, wire, qargs=None, cargs=None):
        """Create a node"""
        self.wire = wire
        super().__init__(qargs, cargs)


class OutNode(DAGNode):
    """Object to represent the information at a node in the DAGCircuit.

    It is used as the return value from `*_nodes()` functions and can
    be supplied to functions that take a node.
    """

    __slots__ = ["wire"]

    def __init__(self, wire, qargs=None, cargs=None):
        """Create a node"""
        self.wire = wire
        super().__init__(qargs, cargs)
