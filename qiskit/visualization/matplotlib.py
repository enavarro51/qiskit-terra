# -*- coding: utf-8 -*-

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

# pylint: disable=invalid-name,missing-docstring,inconsistent-return-statements

"""mpl circuit visualization backend."""

import collections
import fractions
import itertools
import json
import logging
import math
from warnings import warn

import numpy as np

try:
    from matplotlib import get_backend
    from matplotlib import patches
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qiskit.circuit import ControlledGate
from qiskit.visualization.qcstyle import DefaultStyle, BWStyle
from qiskit import user_config
from qiskit.circuit.tools.pi_check import pi_check

logger = logging.getLogger(__name__)

WID = 0.65
HIG = 0.65
DEFAULT_SCALE = 4.3
PORDER_GATE = 5
PORDER_LINE = 3
PORDER_REGLINE = 2
PORDER_GRAY = 3
PORDER_TEXT = 6
PORDER_SUBP = 4


class Anchor:
    def __init__(self, reg_num, yind, fold):
        self.__yind = yind
        self.__fold = fold
        self.__reg_num = reg_num
        self.__gate_placed = []
        self.gate_anchor = 0

    def plot_coord(self, index, gate_width, x_offset):
        h_pos = index % self.__fold + 1
        # check folding
        if self.__fold > 0:
            if h_pos + (gate_width - 1) > self.__fold:
                index += self.__fold - (h_pos - 1)
            x_pos = index % self.__fold + 1 + 0.5 * (gate_width - 1)
            y_pos = self.__yind - (index // self.__fold) * (self.__reg_num + 1)
        else:
            x_pos = index + 1 + 0.5 * (gate_width - 1)
            y_pos = self.__yind

        # could have been updated, so need to store
        self.gate_anchor = index
        return x_pos + x_offset, y_pos

    def is_locatable(self, index, gate_width):
        hold = [index + i for i in range(gate_width)]
        for p in hold:
            if p in self.__gate_placed:
                return False
        return True

    def set_index(self, index, gate_width):
        if self.__fold < 2:
            _index = index
        else:
            h_pos = index % self.__fold + 1
            if h_pos + (gate_width - 1) > self.__fold:
                _index = index + self.__fold - (h_pos - 1) + 1
            else:
                _index = index
        for ii in range(gate_width):
            if _index + ii not in self.__gate_placed:
                self.__gate_placed.append(_index + ii)
        self.__gate_placed.sort()

    def get_index(self):
        if self.__gate_placed:
            return self.__gate_placed[-1] + 1
        return 0


class MatplotlibDrawer:
    def __init__(self, qregs, cregs, ops,
                 scale=1.0, style=None, plot_barriers=True,
                 reverse_bits=False, layout=None, fold=25, ax=None,
                 initial_state=False, cregbundle=True):

        if not HAS_MATPLOTLIB:
            raise ImportError('The class MatplotlibDrawer needs matplotlib. '
                              'To install, run "pip install matplotlib".')

        self._ast = None
        self._scale = DEFAULT_SCALE * scale
        self._creg = []
        self._qreg = []
        self._registers(cregs, qregs)
        self._ops = ops

        self._qreg_dict = collections.OrderedDict()
        self._creg_dict = collections.OrderedDict()
        self._cond = {
            'n_lines': 0,
            'xmax': 0,
            'ymax': 0,
        }
        config = user_config.get_config()
        if config and (style is None):
            config_style = config.get('circuit_mpl_style', 'default')
            if config_style == 'default':
                self._style = DefaultStyle()
            elif config_style == 'bw':
                self._style = BWStyle()
        elif style is False:
            self._style = BWStyle()
        else:
            self._style = DefaultStyle()

        self.plot_barriers = plot_barriers
        self.reverse_bits = reverse_bits
        self.layout = layout
        self.initial_state = initial_state
        if style and 'cregbundle' in style.keys():
            self.cregbundle = style['cregbundle']
            del style['cregbundle']
            warn("The style dictionary key 'cregbundle' has been deprecated and will be removed"
                 " in a future release. cregbundle is now a parameter to draw()."
                 " Example: circuit.draw(output='mpl', cregbundle=False)", DeprecationWarning, 2)
        else:
            self.cregbundle = cregbundle
        if style:
            if isinstance(style, dict):
                self._style.set_style(style)
            elif isinstance(style, str):
                with open(style, 'r') as infile:
                    dic = json.load(infile)
                self._style.set_style(dic)
        if ax is None:
            self.return_fig = True
            self.figure = plt.figure()
            self.figure.patch.set_facecolor(color=self._style.bg)
            self.ax = self.figure.add_subplot(111)
        else:
            self.return_fig = False
            self.ax = ax
            self.figure = ax.get_figure()

        self.x_offset = 0
        self._reg_long_text = 0

        f = plt.figure()
        if hasattr(f.canvas, 'get_renderer'):
            self.renderer = f.canvas.get_renderer()
        else:
            self.renderer = None

        self.fold = fold
        if self.fold < 2:
            self.fold = -1

        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelbottom=False, labeltop=False,
                            labelleft=False, labelright=False)

        self._latex_chars = ('$', '{', '}', '_', '\\left', '\\right', '\\dagger', '\\rangle')
        self._latex_chars1 = ('\\mapsto', '\\pi')
        self._char_list = {' ': (0.0841, 0.0512), '!': (0.106, 0.064), '"': (0.1224, 0.0768),
                           '#': (0.2211, 0.1371), '$': (0.1681, 0.1023), '%': (0.2504, 0.1553),
                           '&': (0.2047, 0.1279), "'": (0.0731, 0.0457), '(': (0.1023, 0.064),
                           ')': (0.1023, 0.064), '*': (0.1316, 0.0822), '+': (0.2193, 0.1371),
                           ',': (0.0841, 0.0512), '-': (0.095, 0.0585), '.': (0.0841, 0.053),
                           '/': (0.0895, 0.0548), '0': (0.1645, 0.1023), '1': (0.1663, 0.1023),
                           '2': (0.1681, 0.1042), '3': (0.1681, 0.1023), '4': (0.1681, 0.1042),
                           '5': (0.1681, 0.1023), '6': (0.1663, 0.1023), '7': (0.1681, 0.1042),
                           '8': (0.1663, 0.1042), '9': (0.1681, 0.1042), ':': (0.0895, 0.053),
                           ';': (0.0895, 0.053), '<': (0.2193, 0.1352), '=': (0.2193, 0.1371),
                           '>': (0.2193, 0.1352), '?': (0.1389, 0.0859), '@': (0.2613, 0.1626),
                           'A': (0.1809, 0.1115), 'B': (0.1791, 0.1115), 'C': (0.1827, 0.1133),
                           'D': (0.2029, 0.1243), 'E': (0.1645, 0.1023), 'F': (0.1499, 0.0932),
                           'G': (0.2029, 0.1261), 'H': (0.1974, 0.1224), 'I': (0.0768, 0.0475),
                           'J': (0.0768, 0.0475), 'K': (0.1718, 0.106), 'L': (0.1462, 0.0914),
                           'M': (0.2266, 0.1407), 'N': (0.1974, 0.1224), 'O': (0.2065, 0.1279),
                           'P': (0.159, 0.0987), 'Q': (0.2065, 0.1279), 'R': (0.1827, 0.1133),
                           'S': (0.1663, 0.1042), 'T': (0.1626, 0.0987), 'U': (0.1937, 0.1188),
                           'V': (0.1809, 0.1115), 'W': (0.2595, 0.1608), 'X': (0.1809, 0.1115),
                           'Y': (0.1608, 0.0987), 'Z': (0.1791, 0.1115), '[': (0.1023, 0.0658),
                           '\\': (0.0895, 0.0548), ']': (0.1023, 0.064), '^': (0.2211, 0.1371),
                           '_': (0.1334, 0.0822), '`': (0.1316, 0.0822), 'a': (0.1626, 0.1005),
                           'b': (0.1681, 0.1023), 'c': (0.1444, 0.0895), 'd': (0.1663, 0.1042),
                           'e': (0.1626, 0.1005), 'f': (0.0914, 0.0585), 'g': (0.1663, 0.1042),
                           'h': (0.1663, 0.1042), 'i': (0.0749, 0.0457), 'j': (0.0749, 0.0457),
                           'k': (0.1517, 0.095), 'l': (0.0749, 0.0457), 'm': (0.2558, 0.159),
                           'n': (0.1663, 0.1042), 'o': (0.1608, 0.0987), 'p': (0.1681, 0.1023),
                           'q': (0.1663, 0.1042), 'r': (0.1096, 0.0676), 's': (0.1371, 0.0841),
                           't': (0.1023, 0.064), 'u': (0.1663, 0.1042), 'v': (0.1553, 0.0969),
                           'w': (0.2156, 0.1334), 'x': (0.1553, 0.0969), 'y': (0.1553, 0.0969),
                           'z': (0.1371, 0.0859), '{': (0.1681, 0.1042), '|': (0.0877, 0.053),
                           '}': (0.1663, 0.1042)}

    def _registers(self, creg, qreg):
        self._creg = []
        for r in creg:
            self._creg.append(r)
        self._qreg = []
        for r in qreg:
            self._qreg.append(r)

    @property
    def ast(self):
        return self._ast

    # This computes the width of a string in the default font
    def _get_text_width(self, text, fontsize):
        if text is None:
            return 0.0

        if self.renderer:
            t = plt.text(0.5, 0.5, text, fontsize=fontsize)
            return t.get_window_extent(renderer=self.renderer).width / 68.4
        else:
            # Remove any latex chars before getting width
            for t in self._latex_chars1:
                text = text.replace(t, 'x')
            for t in self._latex_chars:
                text = text.replace(t, '')

            f = 0 if fontsize == self._style.fs else 1
            return sum([self._char_list[c][f] for c in text])

    def _custom_multiqubit_gate(self, xy, fc=None, text=None, subtext=None):
        xpos = min([x[0] for x in xy])
        ypos = min([y[1] for y in xy])
        ypos_max = max([y[1] for y in xy])

        if text is None:
            gate_text = ''
        elif text in self._style.disptex:
            gate_text = "${}$".format(self._style.disptex[text])
        else:
            gate_text = "${}$".format(text[0].upper()+text[1:])
        text_width = self._get_text_width(gate_text, self._style.fs) + .2
        subtext_width = self._get_text_width(subtext, self._style.sfs) + .2

        if subtext_width > text_width and subtext_width > WID:
            wid = subtext_width
        elif text_width > WID:
            wid = text_width
        else:
            wid = WID

        if self._style.name != 'bw':
            if fc:
                _fc = fc
            elif self._style.gc != DefaultStyle().gc:
                _fc = self._style.gc
            else:
                _fc = self._style.dispcol['multi']
            _ec = _fc
        else:
            _fc = self._style.gc
            _ec = self._style.edge_color

        qubit_span = abs(ypos) - abs(ypos_max) + 1
        height = HIG + (qubit_span - 1)
        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - .5 * HIG), width=wid, height=height,
            fc=_fc, ec=_ec, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        # Annotate inputs
        for bit, y in enumerate([x[1] for x in xy]):
            self.ax.text(xpos - 0.45 * wid, y, str(bit), ha='left', va='center',
                         fontsize=self._style.fs, color=self._style.gt,
                         clip_on=True, zorder=PORDER_TEXT)
        if text:
            if subtext:
                self.ax.text(xpos+.1, ypos + 0.4 * height, gate_text, ha='center',
                             va='center', fontsize=self._style.fs,
                             color=self._style.gt, clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos+.1, ypos + 0.2 * height, subtext, ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.sc, clip_on=True,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos+.1, ypos + .5 * (qubit_span - 1), gate_text,
                             ha='center', va='center', fontsize=self._style.fs,
                             color=self._style.gt, clip_on=True,
                             zorder=PORDER_TEXT, wrap=True)

    def _gate(self, xy, fc=None, text=None, subtext=None):
        xpos, ypos = xy

        if text is None:
            gate_text = ''
        elif text in self._style.disptex:
            gate_text = "${}$".format(self._style.disptex[text])
        else:
            gate_text = "${}$".format(text[0].upper()+text[1:])

        text_width = self._get_text_width(gate_text, self._style.fs)
        subtext_width = self._get_text_width(subtext, self._style.sfs)

        if subtext_width > text_width and subtext_width > WID:
            wid = subtext_width
        elif text_width > WID:
            wid = text_width
        else:
            wid = WID

        if self._style.name != 'bw':
            if fc:
                _fc = fc
            elif self._style.gc != DefaultStyle().gc:
                _fc = self._style.gc
            elif text and text in self._style.dispcol:
                _fc = self._style.dispcol[text]
            else:
                _fc = self._style.dispcol['multi']
            _ec = _fc
        else:
            _fc = self._style.gc
            _ec = self._style.edge_color

        box = patches.Rectangle(xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG),
                                width=wid, height=HIG, fc=_fc, ec=_ec,
                                linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        if text:
            font_size = self._style.fs
            sub_font_size = self._style.sfs

            # check if gate is not unitary
            if text in ['reset']:
                disp_color = self._style.not_gate_lc
                sub_color = self._style.not_gate_lc
            else:
                disp_color = self._style.gt
                sub_color = self._style.sc

            if subtext:
                self.ax.text(xpos, ypos + 0.15 * HIG, gate_text, ha='center',
                             va='center', fontsize=font_size, color=disp_color,
                             clip_on=True, zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos - 0.3 * HIG, subtext, ha='center',
                             va='center', fontsize=sub_font_size, color=sub_color,
                             clip_on=True, zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos, gate_text, ha='center', va='center',
                             fontsize=font_size, color=disp_color,
                             clip_on=True, zorder=PORDER_TEXT)

    def _subtext(self, xy, text):
        xpos, ypos = xy

        self.ax.text(xpos, ypos - 0.3 * HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs, color=self._style.tc,
                     clip_on=True, zorder=PORDER_TEXT)

    def _sidetext(self, xy, text):
        xpos, ypos = xy

        # 0.15 = the initial gap, add 1/2 text width to place on the right
        text_width = self._get_text_width(text, self._style.sfs)
        xp = xpos + 0.15 + text_width / 2
        self.ax.text(xp, ypos + HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs, color=self._style.tc,
                     clip_on=True, zorder=PORDER_TEXT)

    def _line(self, xy0, xy1, lc=None, ls=None, zorder=PORDER_LINE):
        x0, y0 = xy0
        x1, y1 = xy1
        linecolor = self._style.lc if lc is None else lc
        linestyle = 'solid' if ls is None else ls

        if linestyle == 'doublet':
            theta = np.arctan2(np.abs(x1 - x0), np.abs(y1 - y0))
            dx = 0.05 * WID * np.cos(theta)
            dy = 0.05 * WID * np.sin(theta)
            self.ax.plot([x0 + dx, x1 + dx], [y0 + dy, y1 + dy],
                         color=linecolor, linewidth=2,
                         linestyle='solid', zorder=zorder)
            self.ax.plot([x0 - dx, x1 - dx], [y0 - dy, y1 - dy],
                         color=linecolor, linewidth=2,
                         linestyle='solid', zorder=zorder)
        else:
            self.ax.plot([x0, x1], [y0, y1],
                         color=linecolor, linewidth=2,
                         linestyle=linestyle, zorder=zorder)

    def _measure(self, qxy, cxy, cid):
        qx, qy = qxy
        cx, cy = cxy

        # draw gate box
        self._gate(qxy, fc=self._style.dispcol['meas'])

        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7,
                          height=HIG * 0.7, theta1=0, theta2=180, fill=False,
                          ec=self._style.not_gate_lc, linewidth=2, zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([qx, qx + 0.35 * WID], [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.not_gate_lc, linewidth=2, zorder=PORDER_GATE)
        # arrow
        self._line(qxy, [cx, cy + 0.35 * WID], lc=self._style.cc, ls=self._style.cline)
        arrowhead = patches.Polygon(((cx - 0.20 * WID, cy + 0.35 * WID),
                                     (cx + 0.20 * WID, cy + 0.35 * WID),
                                     (cx, cy)), fc=self._style.cc, ec=None)
        self.ax.add_artist(arrowhead)
        # target
        if self.cregbundle:
            self.ax.text(cx + .25, cy + .1, str(cid), ha='left', va='bottom',
                         fontsize=0.8 * self._style.fs, color=self._style.tc,
                         clip_on=True, zorder=PORDER_TEXT)

    def _conds(self, xy, istrue=False):
        xpos, ypos = xy

        fc = self._style.lc if istrue else self._style.gc
        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15, fc=fc,
                             ec=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _ctrl_qubit(self, xy, fc=None, ec=None):
        if fc is None:
            fc = self._style.gc
        if ec is None:
            ec = self._style.lc
        xpos, ypos = xy
        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=fc, ec=ec, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _set_multi_ctrl_bits(self, ctrl_state, num_ctrl_qubits, qbit, color):
        cstate = "{0:b}".format(ctrl_state).rjust(num_ctrl_qubits, '0')[::-1]
        for i in range(num_ctrl_qubits):
            # Make facecolor of ctrl bit the box color if closed and bkgrnd if open
            if self._style.name != 'bw':
                fc_open_close = color if cstate[i] == '1' else self._style.bg
            else:
                fc_open_close = self._style.tc if cstate[i] == '1' else self._style.bg
            ec = color if self._style.name != 'bw' else self._style.lc
            self._ctrl_qubit(qbit[i], fc=fc_open_close, ec=ec)

    def _x_tgt_qubit(self, xy, fc=None, ec=None, ac=None):
        if self._style.gc != DefaultStyle().gc:
            fc = self._style.gt
            ec = self._style.gt
        if fc is None:
            fc = self._style.dispcol['target']
        if ec is None:
            ec = self._style.lc
        if ac is None:
            ac = self._style.lc

        linewidth = 2
        xpos, ypos = xy
        box = patches.Circle(xy=(xpos, ypos), radius=HIG * 0.35,
                             fc=fc, ec=ec, linewidth=linewidth,
                             zorder=PORDER_GATE)
        self.ax.add_patch(box)

        # add '+' symbol
        self.ax.plot([xpos, xpos], [ypos - 0.2 * HIG, ypos + 0.2 * HIG],
                     color=ac, linewidth=linewidth, zorder=PORDER_GATE + 1)

        self.ax.plot([xpos - 0.2 * HIG, xpos + 0.2 * HIG], [ypos, ypos],
                     color=ac, linewidth=linewidth, zorder=PORDER_GATE + 1)

    def _swap(self, xy, color):
        xpos, ypos = xy

        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=color, linewidth=2, zorder=PORDER_LINE + 1)
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos + 0.20 * WID, ypos - 0.20 * WID],
                     color=color, linewidth=2, zorder=PORDER_LINE + 1)

    def _barrier(self, config):
        xys = config['coord']
        group = config['group']
        y_reg = []
        for qreg in self._qreg_dict.values():
            if qreg['group'] in group:
                y_reg.append(qreg['y'])

        for xy in xys:
            xpos, ypos = xy
            self.ax.plot([xpos, xpos], [ypos + 0.5, ypos - 0.5],
                         linewidth=1, linestyle="dashed",
                         color=self._style.lc, zorder=PORDER_TEXT)
            box = patches.Rectangle(xy=(xpos - (0.3 * WID), ypos - 0.5),
                                    width=0.6 * WID, height=1,
                                    fc=self._style.bc, ec=None, alpha=0.6,
                                    linewidth=1.5, zorder=PORDER_GRAY)
            self.ax.add_patch(box)

    def _linefeed_mark(self, xy):
        xpos, ypos = xy

        self.ax.plot([xpos - .1, xpos - .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, zorder=PORDER_LINE)
        self.ax.plot([xpos + .1, xpos + .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, zorder=PORDER_LINE)

    def draw(self, filename=None, verbose=False):
        self._draw_regs()
        self._draw_ops(verbose)
        _xl = - self._style.margin[0]
        _xr = self._cond['xmax'] + self._style.margin[1]
        _yb = - self._cond['ymax'] - self._style.margin[2] + 1 - 0.5
        _yt = self._style.margin[3] + 0.5
        self.ax.set_xlim(_xl, _xr)
        self.ax.set_ylim(_yb, _yt)

        # update figure size
        fig_w = _xr - _xl
        fig_h = _yt - _yb
        if self._style.figwidth < 0.0:
            self._style.figwidth = fig_w * self._scale * self._style.fs / 72 / WID
        self.figure.set_size_inches(self._style.figwidth, self._style.figwidth * fig_h / fig_w)
        self.figure.tight_layout()

        if filename:
            self.figure.savefig(filename, dpi=self._style.dpi,
                                bbox_inches='tight', facecolor=self.figure.get_facecolor())
        if self.return_fig:
            if get_backend() in ['module://ipykernel.pylab.backend_inline',
                                 'nbAgg']:
                plt.close(self.figure)
            return self.figure

    def _draw_regs(self):
        longest_label_width = 0
        if self.initial_state:
            initial_qbit = ' |0>'
            initial_cbit = ' 0'
        else:
            initial_qbit = ''
            initial_cbit = ''

        def _fix_double_script(label):
            words = label.split(' ')
            words = [word.replace('_', r'\_') if word.count('_') > 1 else word
                     for word in words]
            words = [word.replace('^', r'\^{\ }') if word.count('^') > 1 else word
                     for word in words]
            return ' '.join(words)

        # quantum register
        for ii, reg in enumerate(self._qreg):
            if len(self._qreg) > 1:
                if self.layout is None:
                    label = '${{{name}}}_{{{index}}}$'.format(name=reg.register.name,
                                                              index=reg.index) + initial_qbit
                    label = _fix_double_script(label)
                    text_width = self._get_text_width(label, self._style.fs)
                else:
                    label = '${{{name}}}_{{{index}}} \\mapsto {{{physical}}}$'.format(
                        name=self.layout[reg.index].register.name,
                        index=self.layout[reg.index].index, physical=reg.index) + initial_qbit
                    label = _fix_double_script(label)
                    text_width = self._get_text_width(label, self._style.fs)
            else:
                label = '${name}$'.format(name=reg.register.name) + initial_qbit
                label = _fix_double_script(label)
                text_width = self._get_text_width(label, self._style.fs)

            if text_width > longest_label_width:
                longest_label_width = text_width

            pos = -ii
            self._qreg_dict[ii] = {
                'y': pos, 'label': label, 'index': reg.index, 'group': reg.register}
            self._cond['n_lines'] += 1

        # classical register
        if self._creg:
            n_creg = self._creg.copy()
            n_creg.pop(0)
            idx = 0
            y_off = -len(self._qreg)
            for ii, (reg, nreg) in enumerate(itertools.zip_longest(self._creg, n_creg)):
                pos = y_off - idx
                if self.cregbundle:
                    label = '${}$'.format(reg.register.name) + initial_cbit
                    label = _fix_double_script(label)
                    text_width = self._get_text_width(reg.register.name, self._style.fs)
                    if text_width > longest_label_width:
                        longest_label_width = text_width
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index,
                                           'group': reg.register}
                    if not (not nreg or reg.register != nreg.register):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.register.name, reg.index) + initial_cbit
                    label = _fix_double_script(label)
                    text_width = self._get_text_width(reg.register.name, self._style.fs)
                    if text_width > longest_label_width:
                        longest_label_width = text_width
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index,
                                           'group': reg.register}
                self._cond['n_lines'] += 1
                idx += 1

        self._reg_long_text = longest_label_width
        self.x_offset = -1.2 + self._reg_long_text

    def _draw_regs_sub(self, n_fold, feedline_l=False, feedline_r=False):
        # quantum register
        for qreg in self._qreg_dict.values():
            label = qreg['label']
            y = qreg['y'] - n_fold * (self._cond['n_lines'] + 1)
            self.ax.text(self.x_offset - 0.2, y, label, ha='right', va='center',
                         fontsize=1.25 * self._style.fs, color=self._style.tc,
                         clip_on=True, zorder=PORDER_TEXT)
            self._line([self.x_offset + 0.2, y], [self._cond['xmax'], y],
                       zorder=PORDER_REGLINE)

        # classical register
        this_creg_dict = {}
        for creg in self._creg_dict.values():
            label = creg['label']
            y = creg['y'] - n_fold * (self._cond['n_lines'] + 1)
            if y not in this_creg_dict.keys():
                this_creg_dict[y] = {'val': 1, 'label': label}
            else:
                this_creg_dict[y]['val'] += 1
        for y, this_creg in this_creg_dict.items():
            # bundle
            if this_creg['val'] > 1:
                self.ax.plot([self.x_offset + 0.64, self.x_offset + 0.74], [y - .1, y + .1],
                             color=self._style.cc, zorder=PORDER_LINE)
                self.ax.text(self.x_offset+0.54, y + .1, str(this_creg['val']), ha='left',
                             va='bottom', fontsize=0.8 * self._style.fs,
                             color=self._style.tc, clip_on=True, zorder=PORDER_TEXT)
            self.ax.text(self.x_offset - 0.2, y, this_creg['label'], ha='right', va='center',
                         fontsize=1.25 * self._style.fs, color=self._style.tc,
                         clip_on=True, zorder=PORDER_TEXT)
            self._line([self.x_offset + 0.2, y], [self._cond['xmax'], y], lc=self._style.cc,
                       ls=self._style.cline, zorder=PORDER_REGLINE)

        # lf line
        if feedline_r:
            self._linefeed_mark((self.fold + self.x_offset + 1 - 0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))
        if feedline_l:
            self._linefeed_mark((self.x_offset + 0.3,
                                 - n_fold * (self._cond['n_lines'] + 1)))

    def _draw_ops(self, verbose=False):
        _narrow_gates = ['x', 'y', 'z', 'id', 'h', 'r', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz',
                         'rxx', 'ryy', 'rzx', 'u1', 'swap', 'reset']
        _barrier_gates = ['barrier', 'snapshot', 'load', 'save', 'noise']
        _barriers = {'coord': [], 'group': []}

        #
        # generate coordinate manager
        #
        q_anchors = {}
        for key, qreg in self._qreg_dict.items():
            q_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=qreg['y'], fold=self.fold)
        c_anchors = {}
        for key, creg in self._creg_dict.items():
            c_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=creg['y'], fold=self.fold)
        #
        # Draw the ops
        #
        prev_anc = -1
        for layer in self._ops:
            widest_box = 0.0
            #
            # Compute the layer_width for this layer
            #
            for op in layer:
                base_name = None if not hasattr(op.op, 'base_gate') else op.op.base_gate.name

                # Narrow gates are all layer_width 1
                if (op.name in _narrow_gates or (base_name != 'u1' and base_name in _narrow_gates)
                        or op.name in _barrier_gates or op.name == 'measure'):
                    box_width = WID
                    continue

                text_width = self._get_text_width(op.name, fontsize=self._style.fs)
                if (op.type == 'op' and hasattr(op.op, 'params')
                        and not any([isinstance(param, np.ndarray) for param in op.op.params])):
                    param = self.param_parse(op.op.params)
                    param_width = self._get_text_width(param, fontsize=self._style.sfs)

                    if op.name == 'cu1' or op.name == 'rzz' or base_name == 'rzz':
                        tname = 'cu1' if op.name == 'cu1' else 'zz'
                        side_width = self._get_text_width(tname + ' ',
                                                          fontsize=self._style.sfs) + param_width
                        box_width = WID + 0.15 + side_width
                    else:
                        if param_width > text_width and param_width > WID:
                            box_width = param_width
                        elif text_width > WID:
                            box_width = text_width
                        else:
                            box_width = WID
                else:
                    box_width = WID if text_width + 0.2 < WID else text_width + 0.2

                if box_width > widest_box:
                    widest_box = box_width

            layer_width = int(widest_box) + 1
            this_anc = prev_anc + 1
            #
            # Draw the gates in this layer
            #
            for op in layer:
                base_name = None if not hasattr(op.op, 'base_gate') else op.op.base_gate.name
                gate_text = getattr(op.op, 'label', None) or op.name

                # get qreg index
                q_idxs = []
                for qarg in op.qargs:
                    for index, reg in self._qreg_dict.items():
                        if (reg['group'] == qarg.register and
                                reg['index'] == qarg.index):
                            q_idxs.append(index)
                            break

                # get creg index
                c_idxs = []
                for carg in op.cargs:
                    for index, reg in self._creg_dict.items():
                        if (reg['group'] == carg.register and
                                reg['index'] == carg.index):
                            c_idxs.append(index)
                            break

                # Only add the gate to the anchors if it is going to be plotted.
                # This prevents additional blank wires at the end of the line if
                # the last instruction is a barrier type
                if self.plot_barriers or op.name not in _barrier_gates:
                    for ii in q_idxs:
                        q_anchors[ii].set_index(this_anc, layer_width)

                # qreg coordinate
                q_xy = [q_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset)
                        for ii in q_idxs]
                # creg coordinate
                c_xy = [c_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset)
                        for ii in c_idxs]
                # bottom and top point of qreg
                qreg_b = min(q_xy, key=lambda xy: xy[1])
                qreg_t = max(q_xy, key=lambda xy: xy[1])

                # update index based on the value from plotting
                this_anc = q_anchors[q_idxs[0]].gate_anchor

                if verbose:
                    print(op)

                if op.type == 'op' and hasattr(op.op, 'params'):
                    param = self.param_parse(op.op.params)
                else:
                    param = None

                # conditional gate
                if op.condition:
                    c_xy = [c_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset) for
                            ii in self._creg_dict]
                    mask = 0
                    for index, cbit in enumerate(self._creg):
                        if cbit.register == op.condition[0]:
                            mask |= (1 << index)
                    val = op.condition[1]
                    # cbit list to consider
                    fmt_c = '{{:0{}b}}'.format(len(c_xy))
                    cmask = list(fmt_c.format(mask))[::-1]
                    # value
                    fmt_v = '{{:0{}b}}'.format(cmask.count('1'))
                    vlist = list(fmt_v.format(val))[::-1]
                    # plot conditionals
                    v_ind = 0
                    xy_plot = []
                    for xy, m in zip(c_xy, cmask):
                        if m == '1':
                            if xy not in xy_plot:
                                if vlist[v_ind] == '1' or self.cregbundle:
                                    self._conds(xy, istrue=True)
                                else:
                                    self._conds(xy, istrue=False)
                                xy_plot.append(xy)
                            v_ind += 1
                    creg_b = sorted(xy_plot, key=lambda xy: xy[1])[0]
                    self._subtext(creg_b, hex(val))
                    self._line(qreg_t, creg_b, lc=self._style.cc,
                               ls=self._style.cline)
                #
                # draw special gates
                #
                if op.name == 'measure':
                    vv = self._creg_dict[c_idxs[0]]['index']
                    self._measure(q_xy[0], c_xy[0], vv)

                elif op.name == 'reset':
                    self._gate(q_xy[0], text=gate_text, fc=self._style.gt)

                elif op.name in _barrier_gates:
                    _barriers = {'coord': [], 'group': []}
                    for index, qbit in enumerate(q_idxs):
                        q_group = self._qreg_dict[qbit]['group']
                        if q_group not in _barriers['group']:
                            _barriers['group'].append(q_group)
                        _barriers['coord'].append(q_xy[index])
                    if self.plot_barriers:
                        self._barrier(_barriers)

                elif op.name == 'initialize':
                    vec = '[%s]' % param
                    label = getattr(op.op, 'label', None)
                    self._custom_multiqubit_gate(q_xy, text=label or "|psi>", subtext=vec)

                # For gates with ndarray params, don't display the params as subtext
                elif (op.type == 'op' and hasattr(op.op, 'params')
                      and any([isinstance(param, np.ndarray) for param in op.op.params])
                      and not isinstance(op.op, ControlledGate)):
                    color_name = op.name if op.name in self._style.dispcol else 'multi'
                    fc = self._style.dispcol[color_name]
                    self._custom_multiqubit_gate(q_xy, text=gate_text, fc=fc)

                #
                # draw single qubit gates
                #
                elif len(q_xy) == 1:
                    self._gate(q_xy[0], text=gate_text, subtext=str(param))

                #
                # draw controlled and special gates
                #

                # cx gates
                elif isinstance(op.op, ControlledGate) and base_name == 'x':
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    opname = 'cx' if op.name == 'cx' else 'multi'
                    color = self._style.dispcol[opname]
                    ec = color if self._style.name != 'bw' else self._style.lc
                    lc = ec
                    self._set_multi_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits, q_xy, color)
                    self._x_tgt_qubit(q_xy[num_ctrl_qubits], fc=color,
                                      ec=ec, ac=self._style.dispcol['target'])
                    self._line(qreg_b, qreg_t, lc=lc)

                # cz gate
                elif op.name == 'cz':
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    color = self._style.dispcol['cz']
                    ec = color if self._style.name != 'bw' else self._style.lc
                    lc = ec
                    self._set_multi_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits, q_xy, color)
                    self._ctrl_qubit(q_xy[1], fc=color, ec=ec)
                    self._line(qreg_b, qreg_t, lc=lc, zorder=PORDER_LINE + 1)

                # cu1 gate
                elif op.name == 'cu1':
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    color_name = 'multi' if self._style.name != 'bw' else 'cz'
                    color = self._style.dispcol[color_name]
                    ec = color if self._style.name != 'bw' else self._style.lc
                    lc = ec
                    self._set_multi_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits, q_xy, color)
                    self._ctrl_qubit(q_xy[num_ctrl_qubits], fc=color, ec=ec)
                    self._sidetext(qreg_b, text='U1 ({})'.format(param))
                    self._line(qreg_b, qreg_t, lc=lc)

                # rzz gate
                elif op.name == 'rzz':
                    color_name = 'multi' if self._style.name != 'bw' else 'cz'
                    color = self._style.dispcol[color_name]
                    ec = color if self._style.name != 'bw' else self._style.lc
                    lc = ec
                    self._ctrl_qubit(q_xy[0], fc=color, ec=ec)
                    self._ctrl_qubit(q_xy[1], fc=color, ec=ec)
                    self._sidetext(qreg_b, text='zz ({})'.format(param))
                    self._line(qreg_b, qreg_t, lc=lc)

                # controlled rzz gate
                elif op.name != 'rzz' and base_name == 'rzz':
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    color_name = 'multi' if self._style.name != 'bw' else 'cz'
                    color = self._style.dispcol[color_name]
                    ec = color if self._style.name != 'bw' else self._style.lc
                    lc = ec
                    self._set_multi_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits, q_xy, color)
                    self._ctrl_qubit(q_xy[num_ctrl_qubits], fc=color, ec=ec)
                    self._ctrl_qubit(q_xy[num_ctrl_qubits+1], fc=color, ec=ec)
                    self._sidetext(qreg_b, text='zz ({})'.format(param))
                    self._line(qreg_b, qreg_t, lc=lc)

                # swap gate
                elif op.name == 'swap':
                    color = self._style.dispcol['swap']
                    self._swap(q_xy[0], color)
                    self._swap(q_xy[1], color)
                    self._line(qreg_b, qreg_t, lc=color)

                # cswap gate
                elif op.name != 'swap' and base_name == 'swap':
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    color_name = 'multi' if self._style.name != 'bw' else 'cz'
                    color = self._style.dispcol[color_name]
                    lc = color if self._style.name != 'bw' else self._style.lc
                    self._set_multi_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits, q_xy, color)
                    self._swap(q_xy[num_ctrl_qubits], color)
                    self._swap(q_xy[num_ctrl_qubits+1], color)
                    self._line(qreg_b, qreg_t, lc=lc)

                # rxx, ryy, rzx, dcx, iswap
                elif op.name in ['rxx', 'ryy', 'rzx', 'dcx', 'iswap']:
                    self._custom_multiqubit_gate(q_xy, fc=self._style.dispcol[op.name],
                                                 text=gate_text)

                # All other controlled gates
                elif isinstance(op.op, ControlledGate):
                    ctrl_text = getattr(op.op.base_gate, 'label', None) or op.op.base_gate.name
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    num_qargs = len(q_xy) - num_ctrl_qubits
                    opname = 'cy' if op.name == 'cy' else 'multi'

                    color = self._style.dispcol[opname]
                    ec = color if self._style.name != 'bw' else self._style.lc
                    lc = ec
                    self._set_multi_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits, q_xy, color)
                    self._line(qreg_b, qreg_t, lc=lc)
                    if num_qargs == 1:
                        self._gate(q_xy[num_ctrl_qubits], text=ctrl_text, fc=color,
                                   subtext='{}'.format(param))
                    else:
                        self._custom_multiqubit_gate(q_xy[num_ctrl_qubits:], fc=color,
                                                     text=ctrl_text, subtext='{}'.format(param))

                # draw custom multi-qubit gate as final default
                else:
                    self._custom_multiqubit_gate(q_xy, fc=self._style.dispcol['multi'],
                                                 text=gate_text, subtext='{}'.format(param))

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self.plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = -1 if all([op.name in _barrier_gates for op in layer]) else 0

            prev_anc = this_anc + layer_width + barrier_offset - 1
        #
        # adjust window size and draw horizontal lines
        #
        anchors = [q_anchors[ii].get_index() for ii in self._qreg_dict]
        max_anc = max(anchors) if anchors else 0
        n_fold = max(0, max_anc - 1) // self.fold if self.fold > 0 else 0

        # window size
        if max_anc > self.fold > 0:
            self._cond['xmax'] = self.fold + 1 + self.x_offset
            self._cond['ymax'] = (n_fold + 1) * (self._cond['n_lines'] + 1) - 1
        else:
            self._cond['xmax'] = max_anc + 1 + self.x_offset
            self._cond['ymax'] = self._cond['n_lines']

        # add horizontal lines
        for ii in range(n_fold + 1):
            feedline_r = (n_fold > 0 and n_fold > ii)
            feedline_l = (ii > 0)
            self._draw_regs_sub(ii, feedline_l, feedline_r)

        # draw anchor index number
        if self._style.index:
            for ii in range(max_anc):
                if self.fold > 0:
                    x_coord = ii % self.fold + self._reg_long_text - 0.2
                    y_coord = - (ii // self.fold) * (self._cond['n_lines'] + 1) + 0.7
                else:
                    x_coord = ii + self._reg_long_text - 0.2
                    y_coord = 0.7
                self.ax.text(x_coord, y_coord, str(ii + 1), ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.tc, clip_on=True, zorder=PORDER_TEXT)

    @staticmethod
    def param_parse(v):
        # create an empty list to store the parameters in
        param_parts = [None] * len(v)
        for i, e in enumerate(v):
            try:
                param_parts[i] = pi_check(e, output='mpl', ndigits=3)
            except TypeError:
                param_parts[i] = str(e)

            if param_parts[i].startswith('-'):
                param_parts[i] = '$-$' + param_parts[i][1:]

        param_parts = ', '.join(param_parts)
        return param_parts

    @staticmethod
    def format_numeric(val, tol=1e-5):
        if isinstance(val, complex):
            return str(val)
        elif complex(val).imag != 0:
            val = complex(val)
        abs_val = abs(val)
        if math.isclose(abs_val, 0.0, abs_tol=1e-100):
            return '0'
        if math.isclose(math.fmod(abs_val, 1.0),
                        0.0, abs_tol=tol) and 0.5 < abs_val < 9999.5:
            return str(int(val))
        if 0.1 <= abs_val < 100.0:
            return '{:.2f}'.format(val)
        return '{:.1e}'.format(val)

    @staticmethod
    def fraction(val, base=np.pi, n=100, tol=1e-5):
        abs_val = abs(val)
        for i in range(1, n):
            for j in range(1, n):
                if math.isclose(abs_val, i / j * base, rel_tol=tol):
                    if val < 0:
                        i *= -1
                    return fractions.Fraction(i, j)
        return None
