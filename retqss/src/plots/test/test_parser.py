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

import __builtin__
import unittest

from src.common import ERR_UNKNOWN_VAR
from src.exception import PlotScriptException
from src.parser import StateLogParser


class StateLogParserTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.original_open = __builtin__.open
        def custom_open(filename, mode, *args, **kwargs):
            if isinstance(filename, MockedStateLog):
                return filename
            else:
                return cls.original_open(filename, mode, *args, **kwargs)
            
        __builtin__.open = custom_open
        
    
    @classmethod
    def tearDownClass(cls):
        __builtin__.open = cls.original_open
        
    def setUp(self):
        self.state_log = MockedStateLog(['0:x,0.2,0.1\t1:y,0.2,0.1',
                                         'it:3\tft:10.5',
                                         't:0',
                                         'x:0\t1\t2',
                                         'x:1\t3\t4',
                                         'q:1\t0.5\t5\t6',
                                         't:9',
                                         'x:0\t11\t12',
                                         'q:0\t0.5\t13\t14'])
        
        self.state_log2 = MockedStateLog(['0:x,0.2,0.1',
                                          'it:3\tft:10.5',
                                          't:0',
                                          'x:0\t1\t2',
                                          't:2',
                                          'x:0\t7\t7',
                                          't:9',
                                          'x:0\t11\t12',
                                          'q:0\t0.5\t13\t14'])        
        
        self.invalid_log = MockedStateLog(['0:x,0.2,0.1',
                                           'it:3\tft:10.5',
                                           'invalid'])

    def test_parse_first_line(self):
        history = StateLogParser(self.state_log).parse('x')
        
        dqmin = history.get_dqmin()
        self.assertEquals(dqmin, 0.1)
        
    def test_parse_second_line(self):
        history = StateLogParser(self.state_log).parse('x')
        
        initial_time = history.get_initial_time()
        end_time = history.get_end_time()
        
        self.assertEquals(initial_time, 3)
        self.assertEquals(end_time, 10.5)
        
    def test_parse_sections_0(self):
        history = StateLogParser(self.state_log).parse('x')
        
        state = history.get_state()
        self.assertEquals(state, 0)
        self.assertEquals(len(history._sections), 2)
        
        section1 = history._sections[0]
        self.assertEquals(section1.q, [1])
        self.assertEquals(section1.x, [1,2])
        self.assertEquals(section1.quantum, 0.2)
        self.assertEquals(section1.from_t, 3)
        self.assertEquals(section1.to_t, 9)
        
        section2 = history._sections[1]
        self.assertEquals(section2.q, [13,14])
        self.assertEquals(section2.x, [11,12])
        self.assertEquals(section2.quantum, 0.5)
        self.assertEquals(section2.from_t, 9)
        self.assertEquals(section2.to_t, 10.5)
        
    def test_parse_sections_1(self):
        history = StateLogParser(self.state_log).parse('y')
        
        state = history.get_state()
        self.assertEquals(state, 1)
        self.assertEquals(len(history._sections), 1)
        
        section1 = history._sections[0]
        self.assertEquals(section1.q, [5,6])
        self.assertEquals(section1.x, [3,4])
        self.assertEquals(section1.quantum, 0.5)
        self.assertEquals(section1.from_t, 3)
        self.assertEquals(section1.to_t, 10.5)
        
    def test_section_before_initial_time_skipped(self):
        history = StateLogParser(self.state_log2).parse('x')
        
        initial_time = history.get_initial_time()        
        state = history.get_state()
        
        self.assertEquals(state, 0)
        self.assertEquals(len(history._sections), 2)
        
        section1 = history._sections[0]
        self.assertEquals(section1.from_t, initial_time)
        self.assertEquals(section1.to_t, 9)
        
        section2 = history._sections[1]
        self.assertEquals(section2.from_t, 9)
        self.assertEquals(section2.to_t, 10.5)        
        
    def test_invalid_log(self):
        parser = StateLogParser(self.invalid_log)
        self.assertRaises(PlotScriptException, parser.parse, ('x',))
        
    def test_parse_inexistent_state(self):
        parser = StateLogParser(self.state_log)
        with self.assertRaises(PlotScriptException) as cm:
            parser.parse('z')
        self.assertEquals(cm.exception.get_error_code(), ERR_UNKNOWN_VAR)
        

class MockedStateLog(object):
    
    def __init__(self, lines):
        self.lines = lines
        if len(self.lines[-1]) > 0:
            self.lines.append(str())
        self.iterator = iter(self.lines)
    
    def readline(self):
        return self.iterator.next()
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        pass