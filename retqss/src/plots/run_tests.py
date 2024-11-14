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

from test.test_parser import StateLogParserTest
from test.test_state import StateSectionTest
from test.test_utils import LinearRangeTest, PolyEvalTest


def run_tests():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(StateLogParserTest))
    test_suite.addTest(unittest.makeSuite(StateSectionTest))
    test_suite.addTest(unittest.makeSuite(LinearRangeTest))
    test_suite.addTest(unittest.makeSuite(PolyEvalTest))
     
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == '__main__':
    run_tests()