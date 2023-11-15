# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""DAGDependency class for representing non-commutativity in a circuit.
"""

import math
import heapq
from collections import OrderedDict, defaultdict, namedtuple
from typing import Dict, Generator, Any, List

import rustworkx as rx

from qiskit.circuit import QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.bit import Bit
from qiskit.dagcircuit import DAGOpNode
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.circuit.commutation_checker import CommutationChecker


BitLocations = namedtuple("BitLocations", ("index", "registers"))


class DAGDependencyV2():
    """Object to represent a quantum circuit as a Directed Acyclic Graph (DAG)
    via operation dependencies (i.e. lack of commutation).

    The nodes in the graph are operations represented by quantum gates.
    The edges correspond to non-commutation between two operations
    (i.e. a dependency). A directed edge from node A to node B means that
    operation A does not commute with operation B.
    The object's methods allow circuits to be constructed.

    **Example:**

    Bell circuit with no measurement.

    .. parsed-literal::

              ┌───┐
        qr_0: ┤ H ├──■──
              └───┘┌─┴─┐
        qr_1: ─────┤ X ├
                   └───┘

    The dependency DAG for the above circuit is represented by two nodes.
    The first one corresponds to Hadamard gate, the second one to the CNOT gate
    as the gates do not commute there is an edge between the two nodes.

    **Reference:**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
    Exact and practical pattern matching for quantum circuit optimization.
    `arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

    """

    def __init__(self):
        """
        Create an empty DAGDependency.
        """
        # Circuit name
        self.name = None

        # Circuit metadata
        self.metadata = {}

        # Cache of dag op node sort keys
        self._key_cache = {}

        # Set of wires (Register,idx) in the dag
        self._wires = set()

        # Directed multigraph whose nodes are operations(gates) and edges
        # represent non-commutativity between two gates.
        self._multi_graph = rx.PyDAG()

        # Map of qreg/creg name to Register object.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

        # List of Qubit/Clbit wires that the DAG acts on.
        self.qubits: List[Qubit] = []
        self.clbits: List[Clbit] = []

        # Dictionary mapping of Qubit and Clbit instances to a tuple comprised of
        # 0) corresponding index in dag.{qubits,clbits} and
        # 1) a list of Register-int pairs for each Register containing the Bit and
        # its index within that register.
        self._qubit_indices: Dict[Qubit, BitLocations] = {}
        self._clbit_indices: Dict[Clbit, BitLocations] = {}

        self._global_phase = 0
        self._calibrations = defaultdict(dict)

        self._op_names = {}

        self.duration = None
        self.unit = "dt"

        self.comm_checker = CommutationChecker()

    @property
    def global_phase(self):
        """Return the global phase of the circuit."""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle):
        """Set the global phase of the circuit.

        Args:
            angle (float, ParameterExpression)
        """
        from qiskit.circuit.parameterexpression import ParameterExpression

        if isinstance(angle, ParameterExpression):
            self._global_phase = angle
        else:
            # Set the phase to the [0, 2π) interval
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * math.pi)

    @property
    def calibrations(self):
        """Return calibration dictionary.

        The custom pulse definition of a given gate is of the form
        ``{'gate_name': {(qubits, params): schedule}}``.
        """
        return dict(self._calibrations)

    @calibrations.setter
    def calibrations(self, calibrations):
        """Set the circuit calibration data from a dictionary of calibration definition.

        Args:
            calibrations (dict): A dictionary of input in the format
                {'gate_name': {(qubits, gate_params): schedule}}
        """
        self._calibrations = defaultdict(dict, calibrations)

    def to_rustworkx(self):
        """Returns the DAGDependency in rustworkx format."""
        return self._multi_graph

    def size(self):
        """Returns the number of gates in the circuit"""
        return len(self._multi_graph)

    def depth(self):
        """Return the circuit depth.
        Returns:
            int: the circuit depth
        """
        depth = rx.dag_longest_path_length(self._multi_graph)
        return depth if depth >= 0 else 0

    def add_qubits(self, qubits):
        """Add individual qubit wires."""
        if any(not isinstance(qubit, Qubit) for qubit in qubits):
            raise DAGDependencyError("not a Qubit instance.")

        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise DAGDependencyError("duplicate qubits %s" % duplicate_qubits)

        for qubit in qubits:
            self.qubits.append(qubit)
            self._qubit_indices[qubit] = BitLocations(len(self.qubits) - 1, [])
            #self._add_wire(qubit)

    def add_clbits(self, clbits):
        """Add individual clbit wires."""
        if any(not isinstance(clbit, Clbit) for clbit in clbits):
            raise DAGDependencyError("not a Clbit instance.")

        duplicate_clbits = set(self.clbits).intersection(clbits)
        if duplicate_clbits:
            raise DAGDependencyError("duplicate clbits %s" % duplicate_clbits)

        for clbit in clbits:
            self.clbits.append(clbit)
            self._clbit_indices[clbit] = BitLocations(len(self.clbits) - 1, [])
            #self._add_wire(clbit)

    def add_qreg(self, qreg):
        """Add qubits in a quantum register."""
        if not isinstance(qreg, QuantumRegister):
            raise DAGDependencyError("not a QuantumRegister instance.")
        if qreg.name in self.qregs:
            raise DAGDependencyError("duplicate register %s" % qreg.name)
        self.qregs[qreg.name] = qreg
        existing_qubits = set(self.qubits)
        for j in range(qreg.size):
            if qreg[j] in self._qubit_indices:
                self._qubit_indices[qreg[j]].registers.append((qreg, j))
            if qreg[j] not in existing_qubits:
                self.qubits.append(qreg[j])
                self._qubit_indices[qreg[j]] = BitLocations(
                    len(self.qubits) - 1, registers=[(qreg, j)]
                )
                #self._add_wire(qreg[j])

    def add_creg(self, creg):
        """Add clbits in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGDependencyError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGDependencyError("duplicate register %s" % creg.name)
        self.cregs[creg.name] = creg
        existing_clbits = set(self.clbits)
        for j in range(creg.size):
            if creg[j] in self._clbit_indices:
                self._clbit_indices[creg[j]].registers.append((creg, j))
            if creg[j] not in existing_clbits:
                self.clbits.append(creg[j])
                self._clbit_indices[creg[j]] = BitLocations(
                    len(self.clbits) - 1, registers=[(creg, j)]
                )
                #self._add_wire(creg[j])

    def find_bit(self, bit: Bit) -> BitLocations:
        """
        Finds locations in the circuit, by mapping the Qubit and Clbit to positional index
        BitLocations is defined as: BitLocations = namedtuple("BitLocations", ("index", "registers"))

        Args:
            bit (Bit): The bit to locate.

        Returns:
            namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
                contains the index at which the ``Bit`` can be found (in either
                :obj:`~DAGCircuit.qubits`, :obj:`~DAGCircuit.clbits`, depending on its
                type). The second element (``registers``) is a list of ``(register, index)``
                pairs with an entry for each :obj:`~Register` in the circuit which contains the
                :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).

          Raises:
            DAGCircuitError: If the supplied :obj:`~Bit` was of an unknown type.
            DAGCircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.
        """
        try:
            if isinstance(bit, Qubit):
                return self._qubit_indices[bit]
            elif isinstance(bit, Clbit):
                return self._clbit_indices[bit]
            else:
                raise DAGCircuitError(f"Could not locate bit of unknown type: {type(bit)}")
        except KeyError as err:
            raise DAGCircuitError(
                f"Could not locate provided bit: {bit}. Has it been added to the DAGCircuit?"
            ) from err

    def _add_multi_graph_node(self, node):
        """
        Args:
            node (DAGDepNode): considered node.

        Returns:
            _node_id(int): corresponding label to the added node.
        """
        _node_id = self._multi_graph.add_node(node)
        node._node_id = _node_id
        return _node_id

    def get_nodes(self):
        """
        Returns:
            generator(dict): iterator over all the nodes.
        """
        return iter(self._multi_graph.nodes())

    def get_node(self, _node_id):
        """
        Args:
            _node_id (int): label of considered node.

        Returns:
            node: corresponding to the label.
        """
        return self._multi_graph.get_node_data(_node_id)

    def _add_multi_graph_edge(self, src_id, dest_id, data):
        """
        Function to add an edge from given data (dict) between two nodes.

        Args:
            src_id (int): label of the first node.
            dest_id (int): label of the second node.
            data (dict): data contained on the edge.

        """
        self._multi_graph.add_edge(src_id, dest_id, data)

    def get_edges(self, src_id, dest_id):
        """
        Edge enumeration between two nodes through method get_all_edge_data.

        Args:
            src_id (int): label of the first node.
            dest_id (int): label of the second node.

        Returns:
            List: corresponding to all edges between the two nodes.
        """
        return self._multi_graph.get_all_edge_data(src_id, dest_id)

    def get_all_edges(self):
        """
        Enumeration of all edges.

        Returns:
            List: corresponding to the label.
        """

        return [
            (src, dest, data)
            for src_node in self._multi_graph.nodes()
            for (src, dest, data) in self._multi_graph.out_edges(src_node._node_id)
        ]

    def get_in_edges(self, _node_id):
        """
        Enumeration of all incoming edges for a given node.

        Args:
            _node_id (int): label of considered node.

        Returns:
            List: corresponding incoming edges data.
        """
        return self._multi_graph.in_edges(_node_id)

    def get_out_edges(self, _node_id):
        """
        Enumeration of all outgoing edges for a given node.

        Args:
            _node_id (int): label of considered node.

        Returns:
            List: corresponding outgoing edges data.
        """
        return self._multi_graph.out_edges(_node_id)

    # def direct_successors(self, _node_id):
    #     """
    #     Direct successors id of a given node as sorted list.

    #     Args:
    #         _node_id (int): label of considered node.

    #     Returns:
    #         List: direct successors id as a sorted list
    #     """
    #     return sorted(self._multi_graph.adj_direction(_node_id, False).keys())

    # def direct_predecessors(self, _node_id):
    #     """
    #     Direct predecessors id of a given node as sorted list.

    #     Args:
    #         _node_id (int): label of considered node.

    #     Returns:
    #         List: direct predecessors id as a sorted list
    #     """
    #     return sorted(self._multi_graph.adj_direction(_node_id, True).keys())

    def successors(self, _node_id):
        """
        Successors id of a given node as sorted list.

        Args:
            _node_id (int): label of considered node.

        Returns:
            List: all successors id as a sorted list
        """
        #return sorted(list(self._multi_graph.adj_direction(_node_id, False).keys()))
        return self._multi_graph.successors(_node_id)#self._multi_graph.get_node_data(_node_id))

    def predecessors(self, _node_id):
        """
        Predecessors id of a given node as sorted list.

        Args:
            _node_id (int): label of considered node.

        Returns:
            List: all predecessors id as a sorted list
        """
        #return sorted(list(self._multi_graph.adj_direction(_node_id, True).keys()))
        return self._multi_graph.predecessors(self._multi_graph.get_node_data(_node_id))

    def topological_nodes(self):
        """
        Yield nodes in topological order.

        Returns:
            generator(DAGNode): node in topological order.
        """

        def _key(x):
            return x.sort_key

        return iter(rx.lexicographical_topological_sort(self._multi_graph, key=_key))

    def _create_op_node(self, operation, qargs, cargs):
        """Creates a DAGDepNode to the graph and update the edges.

        Args:
            operation (qiskit.circuit.Operation): operation
            qargs (list[~qiskit.circuit.Qubit]): list of qubits on which the operation acts
            cargs (list[Clbit]): list of classical wires to attach to

        Returns:
            DAGDepNode: the newly added node.
        """
        directives = ["measure"]
        if not getattr(operation, "_directive", False) and operation.name not in directives:
            qindices_list = []
            for elem in qargs:
                qindices_list.append(self.qubits.index(elem))

            if getattr(operation, "condition", None):
                # The change to handling operation.condition follows code patterns in quantum_circuit.py.
                # However:
                #   (1) cindices_list are specific to template optimization and should not be computed
                #       in this place.
                #   (2) Template optimization pass needs currently does not handle general conditions.
                cond_bits = condition_resources(operation.condition).clbits
                cindices_list = [self.clbits.index(clbit) for clbit in cond_bits]
            else:
                cindices_list = []
        else:
            qindices_list = []
            cindices_list = []

        new_node = DAGOpNode(
            op=operation,
            qargs=qargs,
            cargs=cargs,
            dag=self,
        )
        return new_node

    def add_op_node(self, operation, qargs, cargs):
        """Add a DAGDepNode to the graph and update the edges.

        Args:
            operation (qiskit.circuit.Operation): operation as a quantum gate
            qargs (list[~qiskit.circuit.Qubit]): list of qubits on which the operation acts
            cargs (list[Clbit]): list of classical wires to attach to
        """
        new_node = self._create_op_node(operation, qargs, cargs)
        self._add_multi_graph_node(new_node)
        self._update_edges()

    def _update_edges(self):
        """
        Updates DagDependency by adding edges to the newly added node (max_node)
        from the previously added nodes.
        For each previously added node (prev_node), an edge from prev_node to max_node
        is added if max_node is "reachable" from prev_node (this means that the two
        nodes can be made adjacent by commuting them with other nodes), but the two nodes
        themselves do not commute.

        Currently. this function is only used when creating a new DAGDependency from another
        representation of a circuit, and hence there are no removed nodes (this is why
        iterating over all nodes is fine).
        """
        max__node_id = len(self._multi_graph) - 1
        max_node = self.get_node(max__node_id)

        reachable = [True] * max__node_id

        # Analyze nodes in the reverse topological order.
        # An improvement to the original algorithm is to consider only direct predecessors
        # and to avoid constructing the lists of forward and backward reachable predecessors
        # for every node when not required.
        for prev__node_id in range(max__node_id - 1, -1, -1):
            if reachable[prev__node_id]:
                prev_node = self.get_node(prev__node_id)

                if not self.comm_checker.commute(
                    prev_node.op,
                    prev_node.qargs,
                    prev_node.cargs,
                    max_node.op,
                    max_node.qargs,
                    max_node.cargs,
                ):
                    # If prev_node and max_node do not commute, then we add an edge
                    # between the two, and mark all direct predecessors of prev_node
                    # as not reaching max_node.
                    self._multi_graph.add_edge(prev__node_id, max__node_id, {"commute": False})

                    predecessor_ids = self._multi_graph.predecessor_indices(prev__node_id)
                    for predecessor_id in predecessor_ids:
                        reachable[predecessor_id] = False
            else:
                # If prev_node cannot reach max_node, then none of its predecessors can
                # reach max_node either.
                predecessor_ids = self._multi_graph.predecessor_indices(prev__node_id)
                for predecessor_id in predecessor_ids:
                    reachable[predecessor_id] = False

    def get_descendants(self, _node_id):
        return rx.descendants(self._multi_graph, _node_id)
        desc_list = []
        print("IN DESC")
        for node in rx.descendants(self._multi_graph, _node_id):
            print(node)
            desc_list.append(self._multi_graph.get_node_data(node))
        return desc_list

    def get_ancestors(self, _node_id):
        anc_list = []
        for node in rx.ancestors(self._multi_graph, _node_id):
            anc_list.append(self._multi_graph.get_node_data(node))
        return anc_list

    def get_successors(self, _node_id):
        return list(self._multi_graph.successors(_node_id))

    def get_predecessors(self, _node_id):
        return list(self._multi_graph.predecessors(_node_id))

    def copy(self):
        """
        Function to copy a DAGDependency object.
        Returns:
            DAGDependency: a copy of a DAGDependency object.
        """

        dag = DAGDependency()
        dag.name = self.name
        dag.cregs = self.cregs.copy()
        dag.qregs = self.qregs.copy()

        for node in self.get_nodes():
            dag._multi_graph.add_node(node.copy())
        for edges in self.get_all_edges():
            dag._multi_graph.add_edge(edges[0], edges[1], edges[2])
        return dag

    def draw(self, scale=0.7, filename=None, style="color"):
        """
        Draws the DAGDependency graph.

        This function needs `pydot <https://github.com/erocarrera/pydot>`, which in turn needs
        Graphviz <https://www.graphviz.org/>` to be installed.

        Args:
            scale (float): scaling factor
            filename (str): file path to save image to (format inferred from name)
            style (str): 'plain': B&W graph
                         'color' (default): color input/output/op nodes

        Returns:
            Ipython.display.Image: if in Jupyter notebook and not saving to file, otherwise None.
        """
        from qiskit.visualization.dag_visualization import dag_drawer

        return dag_drawer(dag=self, scale=scale, filename=filename, style=style)

    def replace_block_with_op(self, node_block, op, wire_pos_map, cycle_check=True):
        """Replace a block of nodes with a single node.

        This is used to consolidate a block of DAGDepNodes into a single
        operation. A typical example is a block of CX and SWAP gates consolidated
        into a LinearFunction. This function is an adaptation of a similar
        function from DAGCircuit.

        It is important that such consolidation preserves commutativity assumptions
        present in DAGDependency. As an example, suppose that every node in a
        block [A, B, C, D] commutes with another node E. Let F be the consolidated
        node, F = A o B o C o D. Then F also commutes with E, and thus the result of
        replacing [A, B, C, D] by F results in a valid DAGDependency. That is, any
        deduction about commutativity in consolidated DAGDependency is correct.
        On the other hand, suppose that at least one of the nodes, say B, does not commute
        with E. Then the consolidated DAGDependency would imply that F does not commute
        with E. Even though F and E may actually commute, it is still safe to assume that
        they do not. That is, the current implementation of consolidation may lead to
        suboptimal but not to incorrect results.

        Args:
            node_block (List[DAGDepNode]): A list of dag nodes that represents the
                node block to be replaced
            op (qiskit.circuit.Operation): The operation to replace the
                block with
            wire_pos_map (Dict[~qiskit.circuit.Qubit, int]): The dictionary mapping the qarg to
                the position. This is necessary to reconstruct the qarg order
                over multiple gates in the combined single op node.
            cycle_check (bool): When set to True this method will check that
                replacing the provided ``node_block`` with a single node
                would introduce a cycle (which would invalidate the
                ``DAGDependency``) and will raise a ``DAGDependencyError`` if a cycle
                would be introduced. This checking comes with a run time
                penalty. If you can guarantee that your input ``node_block`` is
                a contiguous block and won't introduce a cycle when it's
                contracted to a single node, this can be set to ``False`` to
                improve the runtime performance of this method.
        Raises:
            DAGDependencyError: if ``cycle_check`` is set to ``True`` and replacing
                the specified block introduces a cycle or if ``node_block`` is
                empty.
        """
        block_qargs = set()
        block_cargs = set()
        block_ids = [x._node_id for x in node_block]

        # If node block is empty return early
        if not node_block:
            raise DAGDependencyError("Can't replace an empty node_block")

        for nd in node_block:
            block_qargs |= set(nd.qargs)
            block_cargs |= set(nd.cargs)
            cond = getattr(nd.op, "condition", None)
            if cond is not None:
                block_cargs.update(condition_resources(cond).clbits)

        # Create replacement node
        new_node = self._create_op_node(
            op,
            qargs=sorted(block_qargs, key=lambda x: wire_pos_map[x]),
            cargs=sorted(block_cargs, key=lambda x: wire_pos_map[x]),
        )

        try:
            new_node._node_id = self._multi_graph.contract_nodes(
                block_ids, new_node, check_cycle=cycle_check
            )
        except rx.DAGWouldCycle as ex:
            raise DAGDependencyError(
                "Replacing the specified node block would introduce a cycle"
            ) from ex


def merge_no_duplicates(*iterables):
    """Merge K list without duplicate using python heapq ordered merging

    Args:
        *iterables: A list of k sorted lists

    Yields:
        Iterator: List from the merging of the k ones (without duplicates
    """
    last = object()
    for val in heapq.merge(*iterables):
        if val != last:
            last = val
            yield val
