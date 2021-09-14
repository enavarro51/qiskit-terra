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

"""Helper function for converting a circuit to a dag"""

import copy

from qiskit.dagcircuit.dagcircuit import DAGCircuit


def circuit_to_dag(circuit):
    """Build a ``DAGCircuit`` object from a ``QuantumCircuit``.

    Args:
        circuit (QuantumCircuit): the input circuit.

    Return:
        DAGCircuit: the DAG representing the input circuit.

    Example:
        .. jupyter-execute::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.dagcircuit import DAGCircuit
            from qiskit.converters import circuit_to_dag
            from qiskit.visualization import dag_drawer
            %matplotlib inline

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            dag = circuit_to_dag(circ)
            dag_drawer(dag)
    """
    
    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    for instruction, qargs, cargs in circuit.data:
        dagcircuit.apply_operation_back(instruction.copy(), qargs, cargs)

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit

    
    dagcircuit1 = copy.copy(circuit._data_dag)
    for node in dagcircuit1.topological_op_nodes():
        dagcircuit1.substitute_node(node, node.op.copy())
    if dagcircuit != dagcircuit1:
        print("\n\n", dagcircuit.count_ops())
        print("\n", dagcircuit1.count_ops(), "\n")
        for node in dagcircuit.topological_op_nodes():
            if hasattr(node.op, "params"):
                print("\n", node.op.params)
        for node in dagcircuit1.topological_op_nodes():
            if hasattr(node.op, "params"):
                print("\nCIRC1", node.op.params, "\n")
    return dagcircuit
