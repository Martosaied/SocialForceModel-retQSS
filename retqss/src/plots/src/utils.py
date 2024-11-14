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
from src.exception import PlotScriptException

def evaluate_polynomial(coeffs, t0, t):
    return reduce(lambda value, (i,coeff): value + coeff*(t-t0)**i,
                  enumerate(coeffs),
                  0)
    
def linear_range(start, stop, size):
    if size < 2:
        return [start][:size]
    period = (stop-start)/(size-1.)
    t = start
    points = [t]
    for _ in xrange(size-2):
        t += period
        points.append(t)
    points.append(stop)
    return points

def execute_and_raise_on_error(error_code, method, *args):
    try:
        return method(*args)
    except PlotScriptException, e:
        raise e
    except Exception:
        raise PlotScriptException(error_code)
        
        
        
        
        
        