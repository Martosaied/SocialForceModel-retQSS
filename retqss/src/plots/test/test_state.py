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

import unittest

from src.state import StateSection
from src.common import N_POINTS


class StateSectionTest(unittest.TestCase):
    
    def setUp(self):
        q = [0.1, 1, 2]
        x = [0.2, 2, 4, 6]
        quantum = 0.5
        from_t = 1.5
        to_t = 2.5
        
        d = (to_t-from_t)/(N_POINTS-1.)
        t1 = from_t + d
        
        self.q0 = q[0]
        self.q1 = q[0] + q[1]*(t1-from_t) + q[2]*(t1-from_t)**2
        
        self.x0 = x[0]
        self.x1 = x[0] + x[1]*(t1-from_t) + x[2]*(t1-from_t)**2 +\
                  x[3]*(t1-from_t)**3
        
        self.x = x
        self.from_t = from_t
        self.to_t = to_t 
        self.section = StateSection(q=q, x=x, quantum=quantum,
                                    from_t=from_t, to_t=to_t)
        
    def test_time_points(self):
        time_points = self.section.time_points()
        
        self.assertEquals(len(time_points), N_POINTS)
        self.assertEquals(time_points[0], self.from_t)
        self.assertEquals(time_points[-1], self.to_t)

    def test_q_points(self):
        q_points = self.section.q_points()
        
        self.assertEquals(len(q_points), N_POINTS)
        self.assertEquals(q_points[0], self.q0)
        self.assertEquals(q_points[1], self.q1)

    def test_x_points(self):
        x_points = self.section.x_points()
        
        self.assertEquals(len(x_points), N_POINTS)
        self.assertEquals(x_points[0], self.x0)
        self.assertEquals(x_points[1], self.x1)

    def test_err_points(self):
        err_points = self.section.err_points()
        
        self.assertEquals(len(err_points), N_POINTS)
        self.assertEquals(err_points[0], self.q0 - self.x0)
        self.assertEquals(err_points[1], self.q1 - self.x1)
        
    def test_x_coeffs(self):
        x0 = self.section.x_coeff(0)
        x1 = self.section.x_coeff(1)
        x2 = self.section.x_coeff(2)
        x3 = self.section.x_coeff(3)
        
        self.assertEquals(x0, self.x[0])
        self.assertEquals(x1, self.x[1])
        self.assertEquals(x2, self.x[2])
        self.assertEquals(x3, self.x[3])
        self.assertRaises(Exception, self.section.x_coeff, (4,))