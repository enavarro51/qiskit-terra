# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
============================================
Visualizations (:mod:`qiskit.visualization`)
============================================

.. currentmodule:: qiskit.visualization

Counts and State Visualizations
===============================

.. autosummary::
   :toctree: ../stubs/

   plot_histogram
   plot_bloch_vector
   plot_bloch_multivector
   plot_state_city
   plot_state_hinton
   plot_state_paulivec
   plot_state_qsphere

Device Visualizations
=====================

.. autosummary::
   :toctree: ../stubs/

   plot_gate_map
   plot_error_map
   plot_circuit_layout

Circuit Visualizations
======================

.. autosummary::
   :toctree: ../stubs/

   circuit_drawer
   ~qiskit.visualization.circuit.qcstyle.DefaultStyle

DAG Visualizations
==================

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.visualization.plots_visualizations.dag_drawer

Pass Manager Visualizations
===========================

.. autosummary::
   :toctree: ../stubs/

   pass_manager_drawer

Pulse Visualizations
====================

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.visualization.pulse_v2.draw
   ~qiskit.visualization.pulse_v2.IQXStandard
   ~qiskit.visualization.pulse_v2.IQXSimple
   ~qiskit.visualization.pulse_v2.IQXDebugging

Timeline Visualizations
=======================

.. autosummary::
   :toctree: ../stubs/

   timeline_drawer
   ~qiskit.visualization.timeline.draw

Single Qubit State Transition Visualizations
============================================

.. autosummary::
   :toctree: ../stubs/

   visualize_transition

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   VisualizationError
"""

import os
import sys

from .circuit.circuit_visualization import circuit_drawer
from .circuit import text
from .circuit import matplotlib
from .circuit import latex
from .circuit.matplotlib import HAS_MATPLOTLIB

from .plots_visualizations.counts_visualization import plot_histogram
from .plots_visualizations.state_visualization import (plot_state_hinton,
                                                       plot_bloch_vector,
                                                       plot_bloch_multivector,
                                                       plot_state_city,
                                                       plot_state_paulivec,
                                                       plot_state_qsphere,
                                                       state_drawer)
from .plots_visualizations.transition_visualization import visualize_transition
from .plots_visualizations.dag_visualization import dag_drawer
from .plots_visualizations.gate_map import plot_gate_map, plot_circuit_layout, plot_error_map
from .plots_visualizations.pass_manager_visualization import pass_manager_drawer

from .pulse.interpolation import step_wise, linear, cubic_spline
from .pulse.qcstyle import PulseStyle, SchedStyle
from .pulse.pulse_visualization import pulse_drawer
from .pulse_v2 import draw as pulse_drawer_v2

from .timeline import draw as timeline_drawer
from .tools import utils, array
from .tools.array import array_to_latex
from .exceptions import VisualizationError
