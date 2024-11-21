package retQSS_social_force_model

import retQSS;
import retQSS_covid19;
import retQSS_covid19_utils;
import retQSS_covid19_fsm;

function randomBoolean
	input Real trueProbability;
	output Real result;
algorithm
	if random(0.0, 1.0) < trueProbability then
		result := 1.0;
	else
		result := 0.0;
	end if;
end randomBoolean;

function randomRoute
	input Real size;
	input Real zCoord;
	output Real x;
	output Real y;
	output Real z;
	output Real dx;
	output Real dy;
	output Real dz;
algorithm
	if randomBoolean(0.5) == 0.0 then
		x := 0.0 * size;
		dx := 0.99 * size;
	else
		x := 0.99 * size;
		dx := 0.0 * size;
	end if;
	dy := random(0.0, size);
	y := random(0.0, size);
	dz := zCoord;
	z := zCoord;
end randomRoute;

function repulsivePedestrianEffect
	input Real pX1;
	input Real pY1;
	input Real pZ1;
	input Real pX2;
	input Real pY2;
	input Real pZ2;
	input Real vX2;
	input Real vY2;
	input Real vZ2;
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_repulsivePedestrianEffect(pX1, pY1, pZ1, pX2, pY2, pZ2, vX2, vY2, vZ2, x, y, z) annotation(
	Library="social_force_model",
	Include="#include \"retqss_social_force_model.h\"");
end repulsivePedestrianEffect;

function pedestrianAcceleration
	input Real pX;
	input Real pY;
	input Real pZ;
	input Real vX;
	input Real vY;
	input Real vZ;
	input Real targetX;
	input Real targetY;
	input Real targetZ;
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_pedestrianAcceleration(pX, pY, pZ, vX, vY, vZ, targetX, targetY, targetZ, x, y, z) annotation(
	Library="social_force_model",
	Include="#include \"retqss_social_force_model.h\"");
end pedestrianAcceleration;

end retQSS_social_force_model;