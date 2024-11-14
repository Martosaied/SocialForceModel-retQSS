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

from common import ERR_PARSING, ERR_UNKNOWN_VAR
from exception import PlotScriptException
from state import StateHistory, StateSection
from utils import execute_and_raise_on_error


class StateLogParser(object):
    
    def __init__(self, filename):
        self.filename = filename
        
    def _read_first_line(self, _file, history):
        # First line contains state number, DeltaQRel and DeltaQMin.
        var = history.get_var()
        line = _file.readline()
        values = map(lambda v: v.strip(), line.split('\t'))
        all_vars = map(lambda v: v.split(':')[1].split(','), values)
        filtered_vars = filter(lambda (i,_var): var == _var[0],
                               enumerate(all_vars))
        matched_vars = len(filtered_vars)
        if matched_vars == 0:
            raise PlotScriptException(ERR_UNKNOWN_VAR)
        elif matched_vars > 1:
            raise Exception
        
        state, data = filtered_vars[0]
        dqrel, dqmin = map(float, data[1:])
        history.set_state(state)
        history.set_dqrel(dqrel)
        history.set_dqmin(dqmin)
    
    def _read_second_line(self, _file, history):
        # Second line contains the initial and end times.
        line = _file.readline()
        it, ft = line.split('\t')
        
        initial_time = float(it.split(':')[1])
        end_time = float(ft.split(':')[1])
        
        history.set_initial_time(initial_time)
        history.set_end_time(end_time)
        
    def _add_new_section(self, history):
        initial_time = history.get_initial_time()
        # Skip sections happening before the initial time.
        if self.t >= initial_time:
            # Split section if some portion happens before the initial time.
            if self.t0 < initial_time:
                self.t0 = initial_time
                
            section = StateSection(from_t=self.t0, to_t=self.t,
                                   q=self.q, x=self.x,
                                   quantum=self.dq)
    
            history.add_section(section)
        self._clear_values(t0=self.t)
        
    def _clear_values(self, t0=None):
        self.t0 = t0 or 0
        self.q = self.x = None
    
    def _read_body(self, _file, history):
        self.line_no = 0
        self.last_t = 0
        self.t_start = None
        self._clear_values()
        
        finished = False

        while not finished:
            line = _file.readline()
            if len(line) == 0:
                finished = True
                continue
            
            self.line_no += 1
            
            # Parse time
            if line[0] == 't':
                self._read_t(line, history)
            # Parse x
            elif line[0] == 'x':
                self._read_x(line, history)
            # Parse q
            elif line[0] == 'q':
                self._read_q(line, history)
            # Parse start/end of simulation
            elif line[:3] == 'sim':
                self._read_sim(line, history)
            else:
                raise Exception
        
        self.t = history.get_end_time()
        if self.x is not None and self.q is not None:
            self._add_new_section(history)  
    
    def _read_t(self, line, history):
        self.t = float(line.split(':')[1])
        self.last_t = self.line_no
    
    def _read_x(self, line, history):
        tab = line.index('\t')
        i = int(line[2:tab])
        dqrel = history.get_dqrel()
        dqmin = history.get_dqmin()
        if i == history.get_state():
            if self.q is not None and self.x is not None:
                self._add_new_section(history)
            self.x = map(float, line.split('\t')[1:])
            if self.last_t == self.line_no-1:
                self.q = self.x[:-1]
                # TODO: check if this is correct.
                d = abs(self.x[0])*dqrel if dqrel < 1 else abs(self.x[0])/dqrel
                self.dq = max(dqmin, d)
    
    def _read_q(self, line, history):
        tab = line.index('\t')
        i = int(line[2:tab])
        if i != history.get_state():
            return
        values = line.split('\t')
        self.dq = float(values[1])
        self.q = map(float, values[2:])
        
    def _read_sim(self, line, history):
        event = line.split(':')[1].strip()
        if event == 'start':
            self.t_start = self.t
        elif event == 'end':
            if self.t_start is None:
                raise Exception
            history.add_sim_section(self.t_start, self.t)
            self.t_start = None
        else:
            raise Exception
        
    def parse(self, var):
        return execute_and_raise_on_error(ERR_PARSING, self._parse, var)
        
    def _parse(self, var):
        history = StateHistory(var)
        
        with open(self.filename, 'r') as _file:
            self._read_first_line(_file, history)
            self._read_second_line(_file, history)
            self._read_body(_file, history)
            
        return history