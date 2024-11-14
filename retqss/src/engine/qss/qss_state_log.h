/*****************************************************************************

 This file is part of QSS Solver.

 QSS Solver is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 QSS Solver is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with QSS Solver.  If not, see <http://www.gnu.org/licenses/>.

 ******************************************************************************/

/** @file debug.h
 **	@brief This file provides tools for logging and visualizing QSS internals.
 **
 **/

#ifndef DEBUG_H_
#define DEBUG_H_

#include <qss/qss_integrator.h>

#define QSS_STATE_LOG "state_log"


int QSS_stateLog_isFlagEnabled(QSS_simulator simulator);
void QSS_stateLog_init(QSS_simulator simulator);
void QSS_stateLog_log_q(int i, QSS_simulator simulator);
void QSS_stateLog_log_x(int i, QSS_simulator simulator);
void QSS_stateLog_logState(int i, QSS_simulator simulator);
void QSS_stateLog_logStart();
void QSS_stateLog_logEnd();
void QSS_stateLog_setTime(double t);


#endif /* DEBUG_H_ */
