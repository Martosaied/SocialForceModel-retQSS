package retQSS_social_force_model_utils

import retQSS;

/*
  Dump the state of the whole model (states of all particles and volumes) in a CSV line
  This function is implemented in C
*/
function social_force_model_outputCSV
	input Real time;
	input Integer N;
	input Real x[1];
	input Real y[1];
	input Real vx[1];
	input Real vy[1];
	output Boolean status;
	external "C" status=social_force_model_outputCSV(time, N, x) annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end outputCSV;

end retQSS_social_force_model_utils;
