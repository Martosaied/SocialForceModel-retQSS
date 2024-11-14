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

from src.utils import linear_range, evaluate_polynomial


class LinearRangeTest(unittest.TestCase):
    
    def test_simple_range(self):
        points = linear_range(start=1, stop=5, size=5)
        
        self.assertEquals(len(points), 5)
        self.assertEquals(points, range(1,6))
        
    def test_float_range(self):
        points = linear_range(start=1, stop=5, size=4)
        
        self.assertEquals(len(points), 4)
        self.assertAlmostEquals(points[1], 7/3., 6)
        self.assertAlmostEquals(points[2], 11/3., 6)
        
    def test_empty_range(self):
        points = linear_range(start=1, stop=5, size=0)
        
        self.assertEquals(len(points), 0)
        
        
class PolyEvalTest(unittest.TestCase):

    def test_eval_poly(self):
        p = [1, 2, 3]
        p_2 = evaluate_polynomial(p, 0, 2)
        v = p[0] + p[1]*2 + p[2]*2**2
        
        self.assertEquals(p_2, v) 