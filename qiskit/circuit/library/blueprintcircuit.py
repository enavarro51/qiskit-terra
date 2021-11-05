# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Blueprint circuit object."""

from typing import Optional
from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametertable import ParameterTable, ParameterView


class BlueprintCircuit(QuantumCircuit, ABC):
    """Blueprint circuit object.

    In many applications it is necessary to pass around the structure a circuit will have without
    explicitly knowing e.g. its number of qubits, or other missing information. This can be solved
    by having a circuit that knows how to construct itself, once all information is available.

    This class provides an interface for such circuits. Before internal data of the circuit is
    accessed, the ``_build`` method is called. There the configuration of the circuit is checked.
    """

    def __init__(self, *regs, name: Optional[str] = None) -> None:
        """Create a new blueprint circuit.

        The ``_valid`` argument is set to ``False`` to indicate that the circuit has not been built yet.
        """
        super().__init__(*regs, name=name)
        self._qregs = []
        self._cregs = []
        self._qubits = []
        self._qubit_indices = dict()
        self._valid = False
        self._first = True

    @abstractmethod
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration allows the circuit to be built.

        Args:
            raise_on_failure: If True, raise if the configuration is invalid. If False, return
                False if the configuration is invalid.

        Returns:
            True, if the configuration is valid. Otherwise, depending on the value of
            ``raise_on_failure`` an error is raised or False is returned.
        """
        raise NotImplementedError

    @abstractmethod
    def _build(self) -> None:
        """Build the circuit."""
        print('\n build\n', self._valid, id(self._data), self._data)
        if self._valid:
            return

        self._data = []
        # check whether the configuration is valid
        self._check_configuration()
        self._valid = True

    def _invalidate(self) -> None:
        """Invalidate the current circuit build."""
        print('\nbp invalid\n', id(self._data), self._data)
        self._data = []
        self._parameter_table = ParameterTable()
        self.global_phase = 0
        self._valid = False

    @property
    def qregs(self):
        """A list of the quantum registers associated with the circuit."""
        return self._qregs

    @qregs.setter
    def qregs(self, qregs):
        """Set the quantum registers associated with the circuit."""
        self._qregs = []
        self._qubits = []
        self._ancillas = []
        self._qubit_indices = {}

        self.add_register(*qregs)
        print('\n qregs\n', id(self._data), self._data)
        if hasattr(self, '_first') and not self._first:
            self._invalidate()#self._valid = False
        self._first = False

    @property
    def data(self):
        print('\nbp data\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().data

    def decompose(self, gates_to_decompose=None):
        print('\nbp decomp\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().decompose(gates_to_decompose)

    def draw(self, *args, **kwargs):
        print('\nbp draw\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().draw(*args, **kwargs)

    @property
    def num_parameters(self) -> int:
        print('\nbp num param\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().num_parameters

    @property
    def parameters(self) -> ParameterView:
        print('\nbp param\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().parameters

    def qasm(self, formatted=False, filename=None, encoding=None):
        print('\nbp qasm\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().qasm(formatted, filename, encoding)

    def append(self, instruction, qargs=None, cargs=None):
        print('\nbp app\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().append(instruction, qargs, cargs)

    def compose(self, other, qubits=None, clbits=None, front=False, inplace=False, wrap=False):
        print('\nbp comp\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().compose(other, qubits, clbits, front, inplace, wrap)

    def inverse(self):
        print('\nbp inv\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().inverse()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def size(self, *args, **kwargs):
        print('\nbp size\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().size(*args, **kwargs)

    def to_instruction(self, parameter_map=None, label=None):
        print('\nbp to inst\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().to_instruction(parameter_map, label=label)

    def to_gate(self, parameter_map=None, label=None):
        print('\nbp to gate\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().to_gate(parameter_map, label=label)

    def depth(self, *args, **kwargs):
        print('\nbp depth\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().depth(*args, **kwargs)

    def count_ops(self):
        print('\nbp ops\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().count_ops()

    def num_nonlocal_gates(self):
        print('\nbp num nonlocal\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().num_nonlocal_gates()

    def num_connected_components(self, unitary_only=False):
        print('\nbp num conn\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().num_connected_components(unitary_only=unitary_only)

    def copy(self, name=None):
        print('\nbp copy\n', self._valid, id(self._data), self._data)
        if not self._valid:
            self._build()
        return super().copy(name=name)
