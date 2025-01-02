package retQSS_social_force_model

import retQSS;
import retQSS_social_force_model_params;
import retQSS_social_force_model_utils;
import retQSS_social_force_model_types;

function setUpParticles
	input Integer N;
	input Real cellEdgeLength;
	input Integer gridDivisions;
	input Real x[1];
	output Boolean _;
	external "C" _=social_force_model_setUpParticles(N, cellEdgeLength, gridDivisions, x) annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.hh\"");
end setUpParticles;


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
		randomValue2 := 0.1; 
		// randomValue2 := random(0.1, size/3);
		x := randomValue2;
		dx := 1.20 * size;
	else
		randomValue2 := 0.9 * size;
		// randomValue2 := random(size/3 * 2, size);
		x := randomValue2;
		dx := -0.20;
	end if;
	randomValue := random(size*0.25, size*0.75);
	dy := randomValue;
	y := randomValue;
	dz := zCoord;
	z := zCoord;
end randomRoute;

function vectorNorm
	input Real x;
	input Real y;
	input Real z;
	output Real norm;
algorithm
	norm := sqrt(x*x + y*y + z*z);
end vectorNorm;

function desiredDirection
	input Real currentX;
	input Real currentY;
	input Real currentZ;
	input Real targetX;
	input Real targetY;
	input Real targetZ;
	output Real desiredX;
	output Real desiredY;
	output Real desiredZ;
protected
	Real norm;
algorithm
	norm := vectorNorm((targetX - currentX), (targetY - currentY), (targetZ - currentZ));
	desiredX := ((targetX - currentX) / norm);
	desiredY := ((targetY - currentY) / norm);
	desiredZ := currentZ;
end desiredDirection;


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
protected
	Integer index;
	Real currentX;
	Real currentY;
	Real currentZ;
	Real desiredSpeedValue;
	Real desiredX;
	Real desiredY;
	Real desiredZ;
	Real currentVX;
	Real currentVY;
	Real currentVZ;
	Real relaxationTime;
algorithm
	currentX := equationArrayGet(pX, particleID);
	currentY := equationArrayGet(pY, particleID);
	currentZ := equationArrayGet(pZ, particleID);

	// The desired speed is gaussian distributed with mean 1.34 m/s and standard deviation 0.26 m/s
	desiredSpeedValue := arrayGet(desiredSpeed, particleID);

	// The desired direction is given by the difference between the current position and the target position
	(desiredX, desiredY, desiredZ) := desiredDirection(
		currentX, currentY, currentZ,
		targetX, targetY, targetZ
	);

	// // The desired acceleration is the difference between the desired speed and the current speed
	desiredX := desiredX*desiredSpeedValue;
	desiredY := desiredY*desiredSpeedValue;
	desiredZ := desiredZ*desiredSpeedValue;

	// Current velocity
	currentVX := equationArrayGet(vX, particleID);
	currentVY := equationArrayGet(vY, particleID);
	currentVZ := equationArrayGet(vZ, particleID);

	// The acceleration is the difference between the desired acceleration and the current acceleration
	// The acceleration has a relaxation time of 0.5 seconds
	relaxationTime := 1/0.5;
	x := (desiredX - currentVX) * relaxationTime;
	y := (desiredY - currentVY) * relaxationTime;
	z := 0;
end acceleration;

function repulsivePedestrianEffect
	input Real aX;
	input Real aY;
	input Real aZ;
	input Real bX;
	input Real bY;
	input Real bZ;
	input Real bVX;
	input Real bVY;
	input Real bVZ;
	input Real bSpeed;
	input Real targetX;
	input Real targetY;
	output Real x;
	output Real y;
	output Real z;
protected
	Real A;
	Real B;
	Real lambda;
	Real rab;
	Real deltax;
	Real deltay;
	Real distanceab;
	Real normalizedX;
	Real normalizedY;
	Real cos_phi;
	Real desiredX;
	Real desiredY;
	Real desiredZ;
	Real area;
	Real fx;
	Real fy;
algorithm
	rab := PEDESTRIAN_R() * 2;
	A := PEDESTRIAN_A();
	B := PEDESTRIAN_B();
	lambda := PEDESTRIAN_LAMBDA();
	
	deltax := bX - aX;
	deltay := bY - aY;
	distanceab := sqrt(deltax*deltax + deltay*deltay);

	normalizedX := (aX - bX) / distanceab;
	normalizedY := (aY - bY) / distanceab;

	fx := A*exp((rab-distanceab)/B)*normalizedX;
	fy := A*exp((rab-distanceab)/B)*normalizedY;

	(desiredX, desiredY, desiredZ) := desiredDirection(
		aX, aY, aZ,
		targetX, targetY, 0	
	);
	cos_phi := -(normalizedX*desiredX) - (normalizedY*desiredY);
	area := lambda + (1-lambda)*((1+cos_phi)/2);

	x := fx*area;
	y := fy*area;
	z := 0;
end repulsivePedestrianEffect;

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
protected
	Integer index;
	Real totalRepulsiveX;
	Real totalRepulsiveY;
	Real totalRepulsiveZ;
	Integer num_neighbors;
	Real repulsiveX;
	Real repulsiveY;
	Real repulsiveZ;
	Integer i0;
algorithm
	totalRepulsiveX := 0;
	totalRepulsiveY := 0;
	totalRepulsiveZ := 0;

    (num_neighbors, totalRepulsiveX, totalRepulsiveY, totalRepulsiveZ) := 
		particleNeighborhood_forEachParticle_2(
			particleID, "repulsive_pedestrian_effect", 
			targetX, targetY
		);


	// for i0 in 1:300 loop
	// 	if i0 <> particleID then
	// 		(repulsiveX, repulsiveY, repulsiveZ) := repulsivePedestrianEffect(
	// 			equationArrayGet(pX, particleID), equationArrayGet(pY, particleID), equationArrayGet(pZ, particleID), 
	// 			equationArrayGet(pX, i0), equationArrayGet(pY, i0), equationArrayGet(pZ, i0), 
	// 			equationArrayGet(vX, i0), equationArrayGet(vY, i0), equationArrayGet(vZ, i0), 
	// 			equationArrayGet(desiredSpeed, i0), targetX, targetY
	// 		);
	// 		totalRepulsiveX := totalRepulsiveX + repulsiveX;
	// 		totalRepulsiveY := totalRepulsiveY + repulsiveY;
	// 		totalRepulsiveZ := totalRepulsiveZ + repulsiveZ;
	// 	end if;
	// end for;
	x := totalRepulsiveX;
	y := totalRepulsiveY;
	z := totalRepulsiveZ;	
end totalRepulsivePedestrianEffect;

function totalRepulsiveBorderEffect
	input Integer particleID;
	input Real cellEdgeLength;
	input Real pX[1];
	input Real pY[1];
	input Real pZ[1];
	output Real x;
	output Real y;
	output Real z;
protected
	Real A;
	Real B;
	Real R;
	Integer i0;
	Integer isObstacle;
	Real aX;
	Real aY;
	Real borderX;
	Real borderY;
	Real borderZ;
	Real deltay;
	Real deltax;
	Real distanceab;
	Real normalizedY;
	Real normalizedX;
	Real fx;
	Real fy;
	Real totalX;
	Real totalY;
	Integer nextObstacle;
algorithm
	totalX := 0;
	totalY := 0;
	A := BORDER_A();
	B := BORDER_B();
	R := BORDER_R();

	// nextObstacle := particle_nextVolumeID(particleID);

	for i0 in 1:9 loop
	// if nextObstacle <> 0 then
		isObstacle := volume_getProperty(i0, "isObstacle");
		if isObstacle then
			aX := equationArrayGet(pX, particleID);
			aY := equationArrayGet(pY, particleID);

			// Calculate the forces from the centroid to be even from all sides
			(borderX, borderY, borderZ) := volume_centroid(i0);

			deltay := (borderY + cellEdgeLength/2) - aY;
			deltax := (borderX + cellEdgeLength/2) - aX;

			distanceab := sqrt(deltax*deltax + deltay*deltay);

			normalizedY := (aY - borderY - cellEdgeLength/2) / distanceab;
			fy := A*exp((R-distanceab)/B)*normalizedY;

			normalizedX := (aX - borderX - cellEdgeLength/2) / distanceab;
			fx := A*exp((R-distanceab)/B)*normalizedX;

			totalX := totalX + fx;
			totalY := totalY + fy;
			z := 0;
		end if;
	// end if;
	end for;
	x := totalX;
	y := totalY;
	z := 0;
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
	input Real cellEdgeLength;
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
	(wallX, wallY, wallZ) := totalRepulsiveBorderEffect(particleID, cellEdgeLength, pX, pY, pZ);

	resultX := accelerationX + repulsiveX + wallX;
	resultY := accelerationY + repulsiveY + wallY;
	resultZ := accelerationZ + repulsiveZ + wallZ;

	x := resultX;
	y := resultY;
	z := resultZ;
end pedestrianTotalMotivation;


end retQSS_social_force_model;