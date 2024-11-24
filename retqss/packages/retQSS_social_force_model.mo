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

function acceleration
	input Integer particleID;
	input Real desiredSpeed[1];
	input Real pX[1];
	input Real pY[1];
	input Real pZ[1];
	input Real vX[1];
	input Real vY[1];
	input Real vZ[1];
	input Real targetX;
	input Real targetY;
	input Real targetZ;
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_acceleration(particleID, desiredSpeed, pX, pY, pZ, vX, vY, vZ, targetX, targetY, targetZ, x, y, z) annotation(		
	Library="social_force_model",
	Include="#include \"retqss_social_force_model.h\"");
end acceleration;

function totalRepulsivePedestrianEffect
	input Integer particleID;
	input Real pX[1];
	input Real pY[1];
	input Real pZ[1];
	input Real vX[1];
	input Real vY[1];
	input Real vZ[1];
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_totalRepulsivePedestrianEffect(particleID, pX, pY, pZ, vX, vY, vZ, x, y, z) annotation(
	Library="social_force_model",
	Include="#include \"retqss_social_force_model.h\"");
end totalRepulsivePedestrianEffect;

function pedestrianTotalMotivation
	input Integer particleID;
	input Real desiredSpeed[1];
	input Real pX[1];
	input Real pY[1];
	input Real pZ[1];
	input Real vX[1];
	input Real vY[1];
	input Real vZ[1];
	input Real targetX;
	input Real targetY;
	input Real targetZ;
	output Real x;
	output Real y;
	output Real z;
protected
	Integer index;
	Real repulsiveX;
	Real repulsiveY;
	Real repulsiveZ;
	Real accelerationX;
	Real accelerationY;
	Real accelerationZ;
algorithm
	// (repulsiveX, repulsiveY, repulsiveZ) := totalRepulsivePedestrianEffect(particleID, pX, pY, pZ, vX, vY, vZ);
	(accelerationX, accelerationY, accelerationZ) := acceleration(particleID, desiredSpeed, pX, pY, pZ, vX, vY, vZ, targetX, targetY, targetZ);
	// + totalRepulsiveBorderEffect(x, y, z)  + totalAttractivePedestrianEffect(x, y, z);

	// if sqrt(accelerationX * accelerationX + accelerationY * accelerationY) > 1.3 * desiredSpeed then
	// 	accelerationX := accelerationX * desiredSpeed * 1.3 / sqrt(accelerationX * accelerationX + accelerationY * accelerationY);
	// 	accelerationY := accelerationY * desiredSpeed * 1.3/ sqrt(accelerationX * accelerationX + accelerationY * accelerationY);
	// end if;
	x := accelerationX;
	y := accelerationY;
	z := accelerationZ;
end pedestrianTotalMotivation;


end retQSS_social_force_model;