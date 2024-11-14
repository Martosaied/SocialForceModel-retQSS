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
from src.utils import linear_range, evaluate_polynomial
from src.common import N_POINTS


class StateHistory(object):
    
    def __init__(self, var_name):
        self.var_name = var_name
        self._sections = list()
        self._sim_sections = list()
    
    def get_var(self):
        return self.var_name
    
    def get_state(self):
        return self.state

    def get_initial_time(self):
        return self.initial_time
    
    def get_end_time(self):
        return self.end_time
    
    def get_dqrel(self):
        return self.dqrel
    
    def get_dqmin(self):
        return self.dqmin
    
    def get_x_order(self):
        return len(self._sections[0].x)
    
    def set_state(self, state):
        self.state = state
        
    def set_initial_time(self, initial_time):
        self.initial_time = initial_time
        
    def set_end_time(self, end_time):
        self.end_time = end_time
        
    def set_dqrel(self, dqrel):
        self.dqrel = dqrel
        
    def set_dqmin(self, dqmin):
        self.dqmin = dqmin
        
    def add_sim_section(self, start, end):
        self._sim_sections.append((start, end))
    
    def add_section(self, section):
        self._sections.append(section)
        
    def sections(self):
        return iter(self._sections)
    
    def sim_sections(self):
        return iter(self._sim_sections)


class StateSection(object):
    
    def __init__(self, from_t, to_t, q, x, quantum):
        self.from_t = from_t
        self.to_t = to_t
        self.q = q
        self.x = x
        self.quantum = quantum
        
        self._time_points = None
        self._q_points = None
        self._x_points = None
        self._err_points = None
        
    def time_points(self):
        if self._time_points is None:
            self._time_points = linear_range(start=self.from_t, stop=self.to_t,
                                             size=N_POINTS)
        return self._time_points
        
    def q_points(self):
        if self._q_points is None:
            self._q_points = self._compute_points(self.q)
        return self._q_points

    def x_points(self):
        if self._x_points is None:
            self._x_points = self._compute_points(self.x)
        return self._x_points
        
    def err_points(self):
        if self._err_points is None:
            q_points = self.q_points()
            x_points = self.x_points()
            self._err_points = [q_points[k] - x_points[k]
                                for k in xrange(len(q_points))]
        return self._err_points
    
    def x_coeff(self, i):
        if i < 0 or i >= len(self.x):
            raise Exception
        return self.x[i]
        
    def _compute_points(self, coeffs):
        points = list()
        for t in self.time_points():
            point = evaluate_polynomial(coeffs, self.from_t, t)
            points.append(point)
        return points