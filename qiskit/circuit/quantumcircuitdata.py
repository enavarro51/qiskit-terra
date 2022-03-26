# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A wrapper class for the purposes of validating modifications to
QuantumCircuit.data while maintaining the interface of a python list."""

import copy

from collections.abc import MutableSequence

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction


class QuantumCircuitData(MutableSequence):
    """A wrapper class for the purposes of validating modifications to
    QuantumCircuit.data while maintaining the interface of a python list."""

    def __init__(self, circuit):
        self._circuit = circuit

    def __getitem__(self, i):
        """if not isinstance(i, slice) and i < 0:
            i += len(self._circuit._node_idx_map)
        try:
            node = self._circuit._node_idx_map[i]
            ret = (node.op, node.qargs, node.cargs)
            #ret = (self._circuit._node_idx_map[i].op, self._circuit._node_idx_map[i].qargs, self._circuit._node_idx_map[i].cargs)
        except KeyError:
            raise IndexError"""
        node = self._circuit._nodes[i]
        return (node.op, node.qargs, node.cargs)

    def __setitem__(self, key, value):
        instruction, qargs, cargs = value

        if not isinstance(instruction, Instruction) and hasattr(instruction, "to_instruction"):
            instruction = instruction.to_instruction()
        if not isinstance(instruction, Instruction):
            raise CircuitError("object is not an Instruction.")

        expanded_qargs = [self._circuit.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self._circuit.cbit_argument_conversion(carg) for carg in cargs or []]

        broadcast_args = list(instruction.broadcast_arguments(expanded_qargs, expanded_cargs))

        if len(broadcast_args) > 1:
            raise CircuitError(
                "QuantumCircuit.data modification does not support argument broadcasting."
            )

        qargs, cargs = broadcast_args[0]

        self._circuit._check_dups(qargs)
        self._circuit._node_idx_map[key].op = instruction
        self._circuit._node_idx_map[key].qargs = qargs
        self._circuit._node_idx_map[key].cargs = cargs

        self._circuit._update_parameter_table(instruction)

    def insert(self, index, value):
        #self._circuit._nodes.insert(index, None)
        #self[index] = value
        self._circuit._nodes.append(self._circuit._nodes[len(self._circuit._nodes - 1)])
        for idx in range(len(self._circuit._nodes) - 2, index, -1):
            self._circuit._nodes[idx] = self._circuit._nodes[idx - 1]
        self._circuit._nodes[index] = value

    def __delitem__(self, i):
        #if not isinstance(i, slice) and i < 0:
        #    i += len(self._circuit._node_idx_map)
        self._circuit._data.remove_op_node(self._circuit._nodes[i])#self._circuit._node_idx_map[i])
        for idx in range(i, len(self._circuit._nodes) - 1):
            self._circuit._nodes[idx] = self._circuit._nodes[idx + 1]

    def __len__(self):
        return len(self._circuit._node_idx_map)

    def __cast(self, other):
        return other._circuit._data if isinstance(other, QuantumCircuitData) else other

    """def __repr__(self):
        return repr(self._circuit._data)

    def __lt__(self, other):
        return self._circuit._data < self.__cast(other)

    def __le__(self, other):
        return self._circuit._data <= self.__cast(other)"""

    def __eq__(self, other):
        return self == other

    """def __gt__(self, other):
        return self._circuit._data > self.__cast(other)

    def __ge__(self, other):
        return self._circuit._data >= self.__cast(other)

    def __add__(self, other):
        return self._circuit._data + self.__cast(other)

    def __radd__(self, other):
        return self.__cast(other) + self._circuit._data

    def __mul__(self, n):
        return self._circuit._data * n

    def __rmul__(self, n):
        return n * self._circuit._data"""

    def sort(self, *args, **kwargs):
        """In-place stable sort. Accepts arguments of list.sort."""
        self._circuit._data.sort(*args, **kwargs)

    def copy(self):
        """Returns a shallow copy of instruction list."""
        return self._circuit._data.topological_op_nodes()#copy.deepcopy(self._circuit._data)
