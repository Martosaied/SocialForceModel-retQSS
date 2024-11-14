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

from src.common import ERR_DEFAULT
from src.exception import PlotScriptException
from src.parser import StateLogParser
from src.plotter import StateHistoryPlotter


def make_plots(log_path, var_names):
    parser = StateLogParser(log_path)
    plotter = StateHistoryPlotter()
    for var in var_names:
        state_history = parser.parse(var)
        plotter.plot(state_history)
    plotter.show()

def run():
    try:
        log_path = sys.argv[1]
        var_names = map(lambda var: var.strip(), sys.argv[2].split(','))
        make_plots(log_path, var_names)
    except PlotScriptException, e:
        sys.exit(e.get_error_code())
    except Exception:
        sys.exit(ERR_DEFAULT)
        

if __name__ == '__main__':
    run()
