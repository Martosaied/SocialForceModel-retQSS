# This file is part of QSS Solver.
#
# QSS Solver is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# QSS Solver is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with QSS Solver.  If not, see <http://www.gnu.org/licenses/>.
#

import sys

from common import ERR_MATPLOTLIB, ERR_PLOTTING
from utils import execute_and_raise_on_error

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import LinearLocator, MaxNLocator, FormatStrFormatter
except Exception:
    sys.exit(ERR_MATPLOTLIB)


class StateHistoryPlotter(object):
    
    def plot(self, state_history):
        return execute_and_raise_on_error(ERR_PLOTTING, self._plot, state_history)
            
    def _plot(self, state_history):
        var = state_history.get_var()
        state = state_history.get_state()
        
        figure = plt.figure()
        figure.suptitle('Debug plot for state variable $%s$ (internal index: %d)'\
                        % (var, state),
                        fontsize=14)
        
        with PlotGrid(state_history) as grid:
            for section in state_history.sections():
                grid.feed_section(section)
            for sim_section in state_history.sim_sections():
                grid.feed_sim_section(sim_section)

    def show(self):
        plt.show()
        
        
class PlotGrid(object):
    
    # Grid dimensions.
    PLOT_GRID_TOP = 0.95
    PLOT_GRID_LEFT = 0.2
    PLOT_GRID_RIGHT = 0.8
    PLOT_GRID_HSPACE = 0

    @classmethod
    def default_plots(cls):
        return [StatePlot, ErrorPlot, QuantumPlot]
    
    def __init__(self, state_history):
        self.state_history = state_history
        
    def _get_coeffs_plots(self):
        x_order = self.state_history.get_x_order()
        return [CoefficientPlot.at(k) for k in xrange(x_order)]
    
    def _initialize_plots(self, plot_classes):
        self.plots = list()
        for index, plot_class in enumerate(plot_classes):
            parent = self.plots[0].get_axes() if index > 0 else None
            axes = plt.subplot(self.grid[index], sharex=parent)
            plot = plot_class(self.state_history, axes)
            plot.configure()
            self.plots.append(plot)
        # Set x axis format on last plot (i.e., the one at the bottom).
        self.plots[-1].format_x_axis()
        
    def initialize(self):
        coeffs_plots = self._get_coeffs_plots()
        plot_classes = self.default_plots() + coeffs_plots
        self.grid = gridspec.GridSpec(len(plot_classes), 1,
                                      hspace=self.PLOT_GRID_HSPACE,
                                      top=self.PLOT_GRID_TOP,
                                      left=self.PLOT_GRID_LEFT,
                                      right=self.PLOT_GRID_RIGHT)
        self._initialize_plots(plot_classes)
            
    def finalize(self):
        for plot in self.plots:
            plot.add_global_data()
            plot.set_legend()

    def feed_section(self, section):
        for plot in self.plots:
            plot.feed_section(section)

    def feed_sim_section(self, sim_section):
        for plot in self.plots:
            plot.feed_sim_section(sim_section)
            
    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, *args, **kwargs):
        self.finalize()
    
    
class Plot(object):
    
    LEGEND_COLUMNS = 1
    
    def __init__(self, state_history, axes):
        self.state_history = state_history
        self.state = state_history.get_state()
        self.axes = axes
        self.feed_count = 0
        self.sim_section_feed_count = 0
        
    def get_axes(self):
        return self.axes
    
    def configure(self):
        plt.setp(self.axes.get_xticklabels(), visible=False)

        self.axes.grid(which='minor', alpha=0.2)                                
        self.axes.grid(which='major', alpha=0.5)
        
        y_axis = self.axes.get_yaxis()
        y_axis.set_major_locator(MaxNLocator(5, integer=False))
        y_axis.set_major_formatter(FormatStrFormatter('%.5g'))        

        y_ticks = y_axis.get_major_ticks()
        y_ticks[-1].label1.set_visible(False)

    def set_legend(self):
        self.axes.legend(bbox_to_anchor=(1.01, 1),
                         loc=2,
                         fancybox=True,
                         borderaxespad=0.,
                         ncol=self.LEGEND_COLUMNS)
        
    def feed_section(self, section):
        time_points = section.time_points()
        self._feed(section, time_points)
        self.feed_count += 1
        self.prev_section = section
        
    def feed_sim_section(self, sim_section):
        self._feed_sim_section(sim_section)
        self.sim_section_feed_count += 1
        
    def _feed_sim_section(self, sim_section):
        pass

    def add_global_data(self):
        pass
        
    def format_x_axis(self):
        initial_time = self.state_history.get_initial_time()
        end_time = self.state_history.get_end_time()
        
        plt.setp(self.axes.get_xticklabels(), visible=True)
        self.axes.set_xlim([initial_time, end_time])
        self.axes.set_xlabel('$t$')

        x_axis = self.axes.get_xaxis()
        x_axis.set_major_locator(LinearLocator(10))
        x_axis.set_major_formatter(FormatStrFormatter('%.3g'))
        
    def _format_label(self, label, count=None, with_state=True):
        if count is None:
            count = self.feed_count
        if count > 0:
            return None
        return label % self.state if with_state else label
        
    def _feed(self, section):
        raise NotImplementedError


class StatePlot(Plot):
    
    Q_COLOR = '#A6C26E'
    X_COLOR = '#668B8B'
    STEP_COLOR = '#E68B8B'
    
    Q_LABEL = '$q_%d$'
    X_LABEL = '$x_%d$'
    
    POINT_LABEL = 'Sections'
    
    STEP_LABEL = 'Step'
    
    POINT_SIZE = 2
    
    LEGEND_COLUMNS = 2
    
    def _feed(self, section, time_points):
        x_points = section.x_points()
        q_points = section.q_points()
        
        x_label = self._format_label(self.X_LABEL)
        self.axes.plot(time_points, x_points, color=self.X_COLOR, label=x_label)
        
        q_label = self._format_label(self.Q_LABEL)
        self.axes.plot(time_points, q_points, color=self.Q_COLOR, label=q_label)
        
        point_label = self._format_label(self.POINT_LABEL, with_state=False)
        self.axes.plot(time_points[0], x_points[0], color=self.X_COLOR,
                       marker='o', markersize=self.POINT_SIZE)
        self.axes.plot(time_points[-1], x_points[-1], color=self.X_COLOR,
                       marker='o', markersize=self.POINT_SIZE,
                       label=point_label)
        
    def _feed_sim_section(self, sim_section):
        step_label = self._format_label(self.STEP_LABEL,
                                        count=self.sim_section_feed_count,
                                        with_state=False)
        self.axes.axvline(sim_section[0], color=self.STEP_COLOR,
                          linestyle='--', label=step_label)


class ErrorPlot(Plot):
    
    ERR_COLOR = '#5C9A9A'    
    
    ERR_LABEL = '$\mathrm{err}_%d$'
    
    def _feed(self, section, time_points):
        err_points = section.err_points()
        
        err_label = self._format_label(self.ERR_LABEL)
        self.axes.plot(time_points, err_points, color=self.ERR_COLOR, label=err_label)


class StepPlot(Plot):

    def _feed(self, section, time_points):
        current_value = self.get_value(section)
        values = [current_value]*len(time_points)
        
        label = self.get_label()
        label = self._format_label(label)

        color = self.get_color()

        self.axes.plot(time_points, values, color=color, label=label)
        
        # Add vertical lines to join constant sections.
        if self.feed_count > 0:
            t0 = time_points[0]
            prev_value = self.get_value(self.prev_section)
            self.axes.plot((t0, t0), (prev_value, current_value),
                           color=color)
            
    def get_color(self):
        raise NotImplementedError
    
    def get_label(self):
        raise NotImplementedError
    
    def get_value(self, section):
        raise NotImplementedError         


class QuantumPlot(StepPlot):
    
    DQ_COLOR = '#5F9EA0'
    DQ_MIN_COLOR = '#00688B'    
    
    DQ_LABEL = '$\Delta \mathrm{Q}_%d$'
    DQMIN_LABEL = '$\Delta \mathrm{Q}_{\mathrm{min}}$'
    
    def add_global_data(self):
        dqmin = self.state_history.get_dqmin()
        self.axes.axhline(y=dqmin, linestyle='--', color=self.DQ_MIN_COLOR,
                          label=self.DQMIN_LABEL)
        
    def get_color(self):
        return self.DQ_COLOR
    
    def get_label(self):
        return self.DQ_LABEL
    
    def get_value(self, section):
        return section.quantum


class CoefficientPlot(StepPlot):
    
    COEFFS_COLORS = ['#A2CD5A', '#6E8B3D', '#556B2F', '#54632C']    
    
    COEFF_LABEL = '$x_{%%d,%d}$'
    
    @classmethod
    def at(cls, index):
        return lambda state_history, axes: cls(state_history, axes, index)
    
    def __init__(self, state_history, axes, index):
        super(CoefficientPlot, self).__init__(state_history, axes)
        self.index = index
        self.label = self.COEFF_LABEL % self.index
        
        color_index = min(self.index, len(self.COEFFS_COLORS)-1)
        self.color = self.COEFFS_COLORS[color_index]
        
    def get_color(self):
        return self.color
    
    def get_label(self):
        return self.label
    
    def get_value(self, section):
        return section.x_coeff(self.index)
