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

#include <qss/qss_state_log.h>


int QSS_stateLog_isFlagEnabled(QSS_simulator simulator)
{
	return simulator->settings->debug & SD_DBG_StateLog;
}

void QSS_stateLog_init(QSS_simulator simulator)
{
	QSS_data qssData = simulator->data;

	SD_output output = simulator->output;
	double *dQMin = qssData->dQMin;
	double *dQRel = qssData->dQRel;
	double initial_time = qssData->it;
	double end_time = qssData->ft;
	const int states = qssData->states;
	int i, j;
	char *var_name;

	FILE *log = fopen(QSS_STATE_LOG, "w");

	for(i = 0; i < states; ++i)
	{
		// Map state index to output variable index, if possible.
		if(output->nSO[i] > 0)
		{
			j = output->SO[i][0];
			var_name = output->variable[j].name;
		}
		else
			var_name = "?";

		fprintf(log, "%d:%s,%.17g,%.17g%s", i,
				var_name,
				dQRel[i], dQMin[i],
				i < states-1 ? "\t" : "\n");
	}

	fprintf(log, "it:%.17g\tft:%.17g\nt:0\n", initial_time, end_time);

	fclose(log);
}

void QSS_stateLog_log_q(int i, QSS_simulator simulator)
{
	QSS_data qssData = simulator->data;

	const int xOrder = qssData->order;
	const int qOrder = xOrder - 1;
	const int coeffs = xOrder + 1;
	double *q = qssData->q;
	double *lqu = qssData->lqu;
	int j;

	FILE *log = fopen(QSS_STATE_LOG, "a");

	fprintf(log, "q:%d\t%.17g", i, lqu[i]);
	for(j = 0; j <= qOrder; ++j)
		fprintf(log, "\t%.17g", q[i*coeffs + j]);
	fwrite("\n", sizeof(char), 1, log);

	fclose(log);
}

void QSS_stateLog_log_x(int i, QSS_simulator simulator)
{
	QSS_data qssData = simulator->data;

	const int xOrder = qssData->order;
	const int coeffs = xOrder + 1;
	double *x = qssData->x;
	int j;

	FILE *log = fopen(QSS_STATE_LOG, "a");

	fprintf(log, "x:%d", i);
	for(j = 0; j <= xOrder; ++j)
		fprintf(log, "\t%.17g", x[i*coeffs + j]);
	fwrite("\n", sizeof(char), 1, log);

	fclose(log);
}

void QSS_stateLog_logState(int i, QSS_simulator simulator)
{
	QSS_stateLog_log_x(i, simulator);
	QSS_stateLog_log_q(i, simulator);
}

void QSS_stateLog_logStart()
{
	FILE *log = fopen(QSS_STATE_LOG, "a");

	fprintf(log, "sim:start\n");

	fclose(log);
}

void QSS_stateLog_logEnd()
{
	FILE *log = fopen(QSS_STATE_LOG, "a");

	fprintf(log, "sim:end\n");

	fclose(log);
}

void QSS_stateLog_setTime(double t)
{
	FILE *log = fopen(QSS_STATE_LOG, "a");

	fprintf(log, "t:%.17g\n", t);

	fclose(log);
}
