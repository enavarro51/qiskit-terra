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

"""
Template matching in the forward direction, it takes an initial
match, a configuration of qubit and both circuit and template as inputs. The
result is a list of match between the template and the circuit.


**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""

from qiskit.circuit.controlledgate import ControlledGate


class ForwardMatch:
    """
    Object to apply template matching in the forward direction.
    """

    def __init__(
        self, circuit_dag_dep, template_dag_dep, node_id_c, node_id_t, qubits, clbits=None
    ):
        """
        Create a ForwardMatch class with necessary arguments.
        Args:
            circuit_dag_dep (DAGDependencyV2): circuit in the dag dependency form.
            template_dag_dep (DAGDependencyV2): template in the dag dependency form.
            node_id_c (int): index of the first gate matched in the circuit.
            node_id_t (int): index of the first gate matched in the template.
            qubits (list): list of considered qubits in the circuit.
            clbits (list): list of considered clbits in the circuit.
        """

        # The dag dependency representation of the circuit
        self.circuit_dag_dep = circuit_dag_dep

        # The dag dependency representation of the template
        self.template_dag_dep = template_dag_dep

        # List of qubit on which the node of the circuit is acting on
        self.qubits = qubits

        # List of qubit on which the node of the circuit is acting on
        self.clbits = clbits if clbits is not None else []

        # Id of the node in the circuit
        self.node_id_c = node_id_c

        # Id of the node in the template
        self.node_id_t = node_id_t

        # List of match
        self.match = []

        # List of candidates for the forward match
        self.candidates = []

        # List of nodes in circuit which are matched
        self.matched_nodes_list = []

        # Transformation of the qarg indices of the circuit to be adapted to the template indices
        self.qarg_indices = []

        # Transformation of the carg indices of the circuit to be adapted to the template indices
        self.carg_indices = []

        self.successorstovisit = {}
        self.isblocked = {}
        self.matchedwith = {}

    def _init_successors_to_visit(self):
        """
        Initialize the attribute list 'SuccessorsToVisit'
        """
        for i in range(0, self.circuit_dag_dep.size()):
            if i == self.node_id_c:
                node = self.circuit_dag_dep.get_node(i)
                self.successorstovisit[node] = self.circuit_dag_dep.successor_indices(i)

    def _init_matched_with(self):
        """
        Initialize the attribute 'matchedwith' in the template DAG dependency.
        """
        for i in range(0, self.circuit_dag_dep.size()):
            if i == self.node_id_c:
                self.matchedwith[self.circuit_dag_dep.get_node(i)] = [self.node_id_t]
            else:
                self.matchedwith[self.circuit_dag_dep.get_node(i)] = []

        for i in range(0, self.template_dag_dep.size()):
            if i == self.node_id_t:
                self.matchedwith[self.template_dag_dep.get_node(i)] = [self.node_id_c]
            else:
                self.matchedwith[self.template_dag_dep.get_node(i)] = []

    def _init_is_blocked(self):
        """
        Initialize the attribute 'isblocked' in the circuit and template DAG dependency's.
        """
        self.isblocked = {
            self.circuit_dag_dep.get_node(i): False for i in range(0, self.circuit_dag_dep.size())
        }
        self.isblocked.update(
            {
                self.template_dag_dep.get_node(i): False
                for i in range(0, self.template_dag_dep.size())
            }
        )

    def _init_list_match(self):
        """
        Initialize the list of matched nodes between the circuit and the template
        with the first match found.
        """
        self.match.append([self.node_id_t, self.node_id_c])

    def _init_matched_nodes(self):
        """
        Initialize the list of current matched nodes.
        """
        self.matched_nodes_list.append(self.circuit_dag_dep.get_node(self.node_id_c))

    def _find_forward_candidates(self, node_id_t):
        """
        Find the candidate nodes to be matched in the template for a given node.
        Args:
            node_id_t (int): considered node id.
        """
        matches = []

        for i in range(0, len(self.match)):
            matches.append(self.match[i][0])

        pred = matches.copy()
        if len(pred) > 1:
            pred.sort()
        pred.remove(node_id_t)

        node_id_t_succs = self.template_dag_dep.successor_indices(node_id_t)
        if node_id_t_succs:
            maximal_index = node_id_t_succs[-1]
            pred_copy = pred.copy()
            for elem in pred_copy:
                if elem > maximal_index:
                    pred.remove(elem)

        block = []
        for node_id in pred:
            for succ in self.template_dag_dep.successor_indices(node_id):
                if succ not in matches:
                    descs = self.template_dag_dep.descendant_indices(succ)
                    block = block + descs
        self.candidates = list(set(set(node_id_t_succs) - set(matches) - set(block)))

    def _update_qarg_indices(self, qarg):
        """
        Change qubit indices of the current circuit node in order to
        be comparable with the indices of the template qubit list.
        Args:
            qarg (list): list of qubit indices from the circuit for a given node.
        """
        self.qarg_indices = []
        for q in qarg:
            if q in self.qubits:
                self.qarg_indices.append(self.qubits.index(q))
        if len(qarg) != len(self.qarg_indices):
            self.qarg_indices = []

    def _update_carg_indices(self, carg):
        """
        Change clbit indices of the current circuit node in order to
        be comparable with the indices of the template clbit list.
        Args:
            carg (list): list of clbit indices from the circuit for a given node.
        """
        self.carg_indices = []
        if carg:
            for q in carg:
                if q in self.clbits:
                    self.carg_indices.append(self.clbits.index(q))
            if len(carg) != len(self.carg_indices):
                self.carg_indices = []

    def _is_same_op(self, node_circuit, node_template):
        """
        Check if two instructions are the same.
        Args:
            node_circuit (DAGOpNode): node in the circuit.
            node_template (DAGOpNode): node in the template.
        Returns:
            bool: True if the same, False otherwise.
        """
        return node_circuit.op.soft_compare(node_template.op)

    def _is_same_q_conf(self, node_circuit, node_template):
        """
        Check if the qubit configurations are compatible.
        Args:
            node_circuit (DAGOpNode): node in the circuit.
            node_template (DAGOpNode): node in the template.
        Returns:
            bool: True if possible, False otherwise.
        """

        if isinstance(node_circuit.op, ControlledGate):
            c_template = node_template.op.num_ctrl_qubits
            if c_template == 1:
                return self.qarg_indices == self.template_dag_dep.qindices_map[node_template]

            else:
                control_qubits_template = self.template_dag_dep.qindices_map[node_template][
                    :c_template
                ]
                control_qubits_circuit = self.qarg_indices[:c_template]

                if set(control_qubits_circuit) == set(control_qubits_template):
                    target_qubits_template = self.template_dag_dep.qindices_map[node_template][
                        c_template::
                    ]
                    target_qubits_circuit = self.qarg_indices[c_template::]

                    if node_template.op.base_gate.name in [
                        "rxx",
                        "ryy",
                        "rzz",
                        "swap",
                        "iswap",
                        "ms",
                    ]:
                        return set(target_qubits_template) == set(target_qubits_circuit)
                    else:
                        return target_qubits_template == target_qubits_circuit
                else:
                    return False
        else:
            if node_template.op.name in ["rxx", "ryy", "rzz", "swap", "iswap", "ms"]:
                return set(self.qarg_indices) == set(
                    self.template_dag_dep.qindices_map[node_template]
                )
            else:
                return self.qarg_indices == self.template_dag_dep.qindices_map[node_template]

    def _is_same_c_conf(self, node_circuit, node_template):
        """
        Check if the clbit configurations are compatible.
        Args:
            node_circuit (DAGOpNode): node in the circuit.
            node_template (DAGOpNode): node in the template.
        Returns:
            bool: True if possible, False otherwise.
        """
        if getattr(node_circuit.op, "condition", None) and getattr(
            node_template.op, "condition", None
        ):
            if set(self.carg_indices) != set(self.template_dag_dep.cindices_map[node_template]):
                return False
            if (
                getattr(node_circuit.op, "condition", None)[1]
                != getattr(node_template.op, "condition", None)[1]
            ):
                return False
        return True

    def run_forward_match(self):
        """
        Apply the forward match algorithm and returns the list of matches given an initial match
        and a circuit qubit configuration.
        """

        # Initialize the new attributes of the DAGOpNodes of the DAGDependencyV2 object
        self._init_successors_to_visit()
        self._init_matched_with()
        self._init_is_blocked()

        # Initialize the list of matches and the stack of matched nodes (circuit)
        self._init_list_match()
        self._init_matched_nodes()

        # While the list of matched nodes is not empty
        while self.matched_nodes_list:
            first_matched = self.matched_nodes_list.pop(0)
            if not self.successorstovisit[first_matched]:
                continue

            # Get the id and the node of the first successor to visit
            trial_successor_id = self.successorstovisit[first_matched].pop(0)
            trial_successor = self.circuit_dag_dep.get_node(trial_successor_id)

            # Update the matched_nodes_list with new attribute successor to visit and sort the list.
            self.matched_nodes_list.append(first_matched)
            self.matched_nodes_list.sort(key=lambda x: self.successorstovisit[x])

            # If the node is blocked and already matched go to the end
            if self.isblocked[trial_successor] | (self.matchedwith[trial_successor] != []):
                continue

            # Search for potential candidates in the template
            self._find_forward_candidates(self.matchedwith[first_matched][0])

            qarg1 = self.circuit_dag_dep.qindices_map[self.circuit_dag_dep.get_node(trial_successor_id)]
            carg1 = self.circuit_dag_dep.cindices_map[self.circuit_dag_dep.get_node(trial_successor_id)]

            # Update the indices for both qubits and clbits in order to be comparable with  the
            # indices in the template circuit.
            self._update_qarg_indices(qarg1)
            self._update_carg_indices(carg1)

            match = False

            # For loop over the candidates (template) to find a match.
            for i in self.candidates:

                # Break the for loop if a match is found.
                if match:
                    break

                node_circuit = self.circuit_dag_dep.get_node(trial_successor_id)
                node_template = self.template_dag_dep.get_node(i)

                # Compare the indices of qubits and the operation name.
                # Necessary but not sufficient conditions for a match to happen.
                if (
                    len(self.qarg_indices) != len(self.template_dag_dep.qindices_map[node_template])
                    or set(self.qarg_indices)
                    != set(self.template_dag_dep.qindices_map[node_template])
                    or node_circuit.name != node_template.name
                ):
                    continue

                # Check if the qubit, clbit configuration are compatible for a match,
                # also check if the operation are the same.
                if (
                    self._is_same_q_conf(node_circuit, node_template)
                    and self._is_same_c_conf(node_circuit, node_template)
                    and self._is_same_op(node_circuit, node_template)
                ):
                    self.matchedwith[trial_successor] = [i]
                    self.matchedwith[self.template_dag_dep.get_node(i)] = [trial_successor_id]

                    # Append the new match to the list of matches.
                    self.match.append([i, trial_successor_id])

                    # Potential successors to visit (circuit) for a given match.
                    potential = self.circuit_dag_dep.successor_indices(trial_successor_id)

                    # If the potential successors to visit are blocked or matched, it is removed.
                    for potential_id in potential:
                        if self.isblocked[self.circuit_dag_dep.get_node(potential_id)] | (
                            self.matchedwith[self.circuit_dag_dep.get_node(potential_id)] != []
                        ):
                            potential.remove(potential_id)

                    sorted_potential = sorted(potential)

                    #  Update the successor to visit attribute
                    self.successorstovisit[trial_successor] = sorted_potential

                    # Add the updated node to the stack.
                    self.matched_nodes_list.append(trial_successor)
                    self.matched_nodes_list.sort(key=lambda x: self.successorstovisit[x])
                    match = True
                    continue

            # If no match is found, block the node and all the successors.
            if not match:
                self.isblocked[trial_successor] = True
                for desc in self.circuit_dag_dep.descendant_indices(trial_successor_id):
                    self.isblocked[self.circuit_dag_dep.get_node(desc)] = True
                    if self.matchedwith[self.circuit_dag_dep.get_node(desc)]:
                        self.match.remove(
                            [self.matchedwith[self.circuit_dag_dep.get_node(desc)][0], desc]
                        )
                        match_id = self.matchedwith[self.circuit_dag_dep.get_node(desc)][0]
                        self.matchedwith[self.template_dag_dep.get_node(match_id)] = []
                        self.matchedwith[self.circuit_dag_dep.get_node(desc)] = []
        return (self.matchedwith, self.isblocked)
