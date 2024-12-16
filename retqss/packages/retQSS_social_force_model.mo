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
protected
	Real randomValue;
	Real randomValue2;
algorithm
	if randomBoolean(0.5) == 0.0 then
		randomValue2 := random(0.1, size/3);
		x := randomValue2;
		dx := 1.20 * size;
	else
		randomValue2 := random(size/3 * 2, size);
		x := randomValue2;
		dx := -0.20;
	end if;
	randomValue := random(size*0.2, size*0.8);
	dy := randomValue;
	y := randomValue;
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
	input Real desiredSpeed[1];
	input Real pX[1];
	input Real pY[1];
	input Real pZ[1];
	input Real vX[1];
	input Real vY[1];
	input Real vZ[1];
	input Real targetX;
	input Real targetY;
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_totalRepulsivePedestrianEffect(particleID, desiredSpeed, pX, pY, pZ, vX, vY, vZ, targetX, targetY, x, y, z) annotation(
	Library="social_force_model",
	Include="#include \"retqss_social_force_model.h\"");
end totalRepulsivePedestrianEffect;

function totalRepulsiveBorderEffect
	input Integer particleID;
	input Real pX[1];
	input Real pY[1];
	input Real pZ[1];
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_totalRepulsiveBorderEffect(particleID, pX, pY, pZ, x, y, z) annotation(
	Library="social_force_model",
	Include="#include \"retqss_social_force_model.h\"");
end totalRepulsiveBorderEffect;

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
	Real desiredSpeedValue;
	Real currentVX;
	Real currentVY;
	Real currentVZ;
	Real wallX;
	Real wallY;
	Real wallZ;
	Real repulsiveX;
	Real repulsiveY;
	Real repulsiveZ;
	Real accelerationX;
	Real accelerationY;
	Real accelerationZ;
	Real resultX;
	Real resultY;
	Real resultZ;
algorithm
	(accelerationX, accelerationY, accelerationZ) := acceleration(particleID, desiredSpeed, pX, pY, pZ, vX, vY, vZ, targetX, targetY, targetZ);
	(repulsiveX, repulsiveY, repulsiveZ) := totalRepulsivePedestrianEffect(particleID, desiredSpeed, pX, pY, pZ, vX, vY, vZ, targetX, targetY);
	(wallX, wallY, wallZ) := totalRepulsiveBorderEffect(particleID, pX, pY, pZ);

	resultX := repulsiveX + accelerationX + wallX;
	resultY := repulsiveY + accelerationY + wallY;
	resultZ := repulsiveZ + accelerationZ + wallZ;

	x := resultX;
	y := resultY;
	z := resultZ;
end pedestrianTotalMotivation;


end retQSS_social_force_model;