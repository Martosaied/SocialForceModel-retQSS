package retQSS_social_force_model

import retQSS;
import retQSS_utils;
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
	    Include="#include \"retqss_social_force_model.h\"");
end setUpParticles;

function setUpWalls
	output Boolean _;
	external "C" _=social_force_model_setUpWalls() annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end setUpWalls;

function setParameters
	output Boolean _;
	external "C" _=social_force_model_setParameters() annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end setParameters;

function internalRepulsiveBorderEffect
	input Real A;
	input Real B;
	input Real R;
	input Integer particleID;
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_repulsiveBorderEffect(A, B, R, particleID, x, y, z) annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end internalRepulsiveBorderEffect;

function neighborsRepulsiveBorderEffect
	input Real A;
	input Real B;
	input Real R;
	input Integer particleID;
	input Real cellEdgeLength;
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_neighborsRepulsiveBorderEffect(A, B, R, particleID, cellEdgeLength, x, y, z) annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end neighborsRepulsiveBorderEffect;

function updateNeighboringVolumes
	input Integer particleID;
	input Integer gridDivisions;
	output Boolean _;
external "C" social_force_model_updateNeighboringVolumes(particleID, gridDivisions) annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end updateNeighboringVolumes;

function volumeBasedRepulsivePedestrianEffect
	input Integer particleID;
	input Real targetX;
	input Real targetY;
	output Real x;
	output Real y;
	output Real z;
external "C" social_force_model_volumeBasedRepulsivePedestrianEffect(particleID, targetX, targetY, x, y, z) annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end volumeBasedRepulsivePedestrianEffect;

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
	input Real fromY;
	input Real toY;
	output Integer group;
	output Real x;
	output Real y;
	output Real z;
	output Real dx;
	output Real dy;
	output Real dz;
protected
	Real randomValue;
	Real randomValue2;
	Real destination;
	Integer left;
	Integer right;
algorithm
	destination := PEDESTRIAN_DESTINATION();

	left := LEFT();
	right := RIGHT();

	if randomBoolean(0.5) == 0.0 then
		// randomValue2 := random(0.1, size/3);
		randomValue2 := 0.1 * size;
		// randomValue2 := random(0, size);
		x := randomValue2;
		dx := random(10, 12.5);
		group := left;
	else
		// randomValue2 := random(size/3 * 2, size);
		randomValue2 := 0.9 * size;
		// randomValue2 := random(0, size);
		x := randomValue2;
		dx := random(40, 42.5);
		group := right;
	end if;
	randomValue := random(fromY, toY);
	if destination == 0.0 then
		dy := randomValue;
	end if;
	if destination == 1.0 then
		dy := size - randomValue;
	end if;
	if destination == 2.0 then
		dy := random(fromY, toY);
	end if;
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
	desiredX := ((targetX - currentX));
	desiredY := ((targetY - currentY));
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
	input Real pedestrianA1;
	input Real pedestrianB1;
	input Real pedestrianA2;
	input Real pedestrianB2;
	input Real pedestrianR;
	input Real pedestrianLambda;
	input Real aX;
	input Real aY;
	input Real aZ;
	input Real bX;
	input Real bY;
	input Real bZ;
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
	Real fx_1;
	Real fy_1;
	Real fx_2;
	Real fy_2;
algorithm
	rab := pedestrianR * 2;
	lambda := pedestrianLambda;
	
	deltax := bX - aX;
	deltay := bY - aY;
	distanceab := sqrt(deltax*deltax + deltay*deltay);

	normalizedX := (aX - bX) / distanceab;
	normalizedY := (aY - bY) / distanceab;

	fx_1 := pedestrianA1*exp((rab-distanceab)/pedestrianB1)*normalizedX;
	fy_1 := pedestrianA1*exp((rab-distanceab)/pedestrianB1)*normalizedY;
	fx_2 := pedestrianA2*exp((rab-distanceab)/pedestrianB2)*normalizedX;
	fy_2 := pedestrianA2*exp((rab-distanceab)/pedestrianB2)*normalizedY;

	(desiredX, desiredY, desiredZ) := desiredDirection(
		aX, aY, aZ,
		targetX, targetY, 0	
	);
	cos_phi := -(normalizedX*desiredX) - (normalizedY*desiredY);
	area := lambda + (1-lambda)*((1+cos_phi)/2);

	x := fx_1*area + fx_2;
	y := fy_1*area + fy_2;
	z := 0;
end repulsivePedestrianEffect;

function totalRepulsivePedestrianEffect
	input Integer totalNumberOfParticles;
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
	Real pedestrianA1;
	Real pedestrianB1;
	Real pedestrianA2;
	Real pedestrianB2;
	Real pedestrianR;
	Real pedestrianLambda;
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
	if PEDESTRIAN_IMPLEMENTATION() == 2 then
		totalRepulsiveX := 0;
		totalRepulsiveY := 0;
		totalRepulsiveZ := 0;

		(totalRepulsiveX, totalRepulsiveY, totalRepulsiveZ) := volumeBasedRepulsivePedestrianEffect(
			particleID,
			targetX,
			targetY
		);
	end if;

	if PEDESTRIAN_IMPLEMENTATION() == 1 then
		totalRepulsiveX := 0;
		totalRepulsiveY := 0;
		totalRepulsiveZ := 0;

    	(num_neighbors, totalRepulsiveX, totalRepulsiveY, totalRepulsiveZ) := 
			particleNeighborhood_forEachParticle_2(
				particleID, "repulsive_pedestrian_effect", 
				targetX, targetY
			);
	end if;

	if PEDESTRIAN_IMPLEMENTATION() == 0 then
		totalRepulsiveX := 0;
		totalRepulsiveY := 0;
		totalRepulsiveZ := 0;
		pedestrianA1 := PEDESTRIAN_A_1();
		pedestrianB1 := PEDESTRIAN_B_1();
		pedestrianA2 := PEDESTRIAN_A_2();
		pedestrianB2 := PEDESTRIAN_B_2();
		pedestrianR := PEDESTRIAN_R();
		pedestrianLambda := PEDESTRIAN_LAMBDA();

		for i0 in 1:10000 loop
			if i0 < totalNumberOfParticles and i0 <> particleID then
				(repulsiveX, repulsiveY, repulsiveZ) := repulsivePedestrianEffect(
					pedestrianA1, pedestrianB1, pedestrianA2, pedestrianB2, pedestrianR, pedestrianLambda,
					equationArrayGet(pX, particleID), equationArrayGet(pY, particleID), equationArrayGet(pZ, particleID), 
					equationArrayGet(pX, i0), equationArrayGet(pY, i0), equationArrayGet(pZ, i0), 
					equationArrayGet(desiredSpeed, i0), targetX, targetY
				);
				totalRepulsiveX := totalRepulsiveX + repulsiveX;
				totalRepulsiveY := totalRepulsiveY + repulsiveY;
				totalRepulsiveZ := totalRepulsiveZ + repulsiveZ;
			end if;
		end for;
	end if;

	x := totalRepulsiveX;
	y := totalRepulsiveY;
	z := totalRepulsiveZ;	
end totalRepulsivePedestrianEffect;

function nearestPointOnVolume
	input Integer volumeID;
	input Real pointX;
	input Real pointY;
	input Real radius;
	output Real x;
	output Real y;
	output Real z;
protected
	Real borderX;
	Real borderY;
	Real borderZ;
	Real m;
	Real b;
	Real h;
	Real k;
	Real r;
	Real A;
	Real B;
	Real C;
	Real D;
	Real sqrt_D;
	Real x1;
	Real x2;
	Real y1;
	Real y2;
	Real distance1;
	Real distance2;
algorithm
	(borderX, borderY, borderZ) := volume_centroid(volumeID);

	m := (borderY - pointY) / (borderX - pointX);
	b := borderY - m * borderX;
	h := borderX;
	k := borderY;
	r := radius;

	A := 1 + m*m;
	B := 2 * m * (b - k) - 2 * h;
	C := h*h + (b - k)*(b - k) - r*r;

	// Discriminante
	D := B*B - 4 * A * C;

	sqrt_D := sqrt(D);
	x1 := (-B + sqrt_D) / (2 * A);
	x2 := (-B - sqrt_D) / (2 * A);
	y1 := m * x1 + b;
	y2 := m * x2 + b;

	// Calculate the distance to the point
	distance1 := sqrt((x1 - pointX)*(x1 - pointX) + (y1 - pointY)*(y1 - pointY));
	distance2 := sqrt((x2 - pointX)*(x2 - pointX) + (y2 - pointY)*(y2 - pointY));

	// Choose the nearest point
	if distance1 < distance2 then
		x := x1;
		y := y1;
	else
		x := x2;
		y := y2;
	end if;

	z := 0;
end nearestPointOnVolume;

function squareNearestPointOnVolume
	input Integer volumeID;
	input Real pointX;
	input Real pointY;
	input Real radius;
	output Real x;
	output Real y;
	output Real z;
protected
	Real borderX;
	Real borderY;
	Real borderZ;
	Real qx;
	Real qy;
	Real f;
	Real intersectX;
	Real intersectY;
algorithm
	(borderX, borderY, borderZ) := volume_centroid(volumeID);

    qx := (pointX - borderX) / radius;
    qy := (pointY - borderY) / radius;

	if abs(qx) > abs(qy) then
		f := abs(qx);
	else
		f := abs(qy);
	end if;

    intersectX := qx/f;
    intersectY := qy/f;
    x := intersectX * radius + borderX;
    y := intersectY * radius + borderY;
    z := 0;
end squareNearestPointOnVolume;

function repulsiveBorderEffect
	input Integer particleID;
	input Real cellEdgeLength;
	input Integer nextObstacle;
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
	Integer isObstacle;
	Real aX;
	Real aY;
	Real aZ;
	Real vX;
	Real vY;
	Real vZ;
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
	Real correction;
algorithm
	A := BORDER_A();
	B := BORDER_B();
	R := BORDER_R();
	totalX := 0;
	totalY := 0;
	if nextObstacle <> 0 then
		isObstacle := volume_getProperty(nextObstacle, "isObstacle");
		if isObstacle then
			(aX, aY, aZ) := particle_currentPosition(particleID);
			(vX, vY, vZ) := particle_currentVelocity(particleID);

			(borderX, borderY, borderZ) := squareNearestPointOnVolume(nextObstacle, aX, aY, cellEdgeLength/2);
			distanceab := volume_distanceToPoint(nextObstacle, aX, aY, aZ);

			normalizedY := (aY - borderY) / distanceab;
			fy := A*exp((R-distanceab)/B)*normalizedY;

			normalizedX := (aX - borderX) / distanceab;
			fx := A*exp((R-distanceab)/B)*normalizedX;

			totalX := fx;
			totalY := fy;
		end if;
	end if;

	x := totalX;
	y := totalY;
	z := 0;
end repulsiveBorderEffect;

function totalRepulsiveBorderEffect
	input Integer totalNumberOfVolumes;
	input Integer particleID;
	input Real cellEdgeLength;
	input Real pX[1];
	input Real pY[1];
	input Real pZ[1];
	output Real x;
	output Real y;
	output Real z;
protected
	Real totalX;
	Real totalY;
	Real totalZ;
	Real repulsiveX;
	Real repulsiveY;
	Real repulsiveZ;
	Integer nextObstacle;
	Integer i0;
	Real A;
	Real B;
	Real R;
algorithm
	totalX := 0;
	totalY := 0;
	totalZ := 0;

	if BORDER_IMPLEMENTATION() == 0 then
		for i0 in 1:10000 loop
			if i0 < totalNumberOfVolumes then
				(repulsiveX, repulsiveY, repulsiveZ) := repulsiveBorderEffect(particleID, cellEdgeLength, i0, pX, pY, pZ);
				totalX := totalX + repulsiveX;
				totalY := totalY + repulsiveY;
				totalZ := totalZ + repulsiveZ;
			end if;
		end for;
	end if;

	if BORDER_IMPLEMENTATION() == 1 then
		nextObstacle := particle_nextVolumeID(particleID);
		if nextObstacle <> 0 then
			(repulsiveX, repulsiveY, repulsiveZ) := repulsiveBorderEffect(particleID, cellEdgeLength, nextObstacle, pX, pY, pZ);
			totalX := repulsiveX;
			totalY := repulsiveY;
			totalZ := repulsiveZ;
		end if;
	end if;


	if BORDER_IMPLEMENTATION() == 2 then
		A := BORDER_A();
		B := BORDER_B();
		R := BORDER_R();
		(totalX, totalY, totalZ) := internalRepulsiveBorderEffect(A, B, R, particleID);
	end if;

	if BORDER_IMPLEMENTATION() == 3 then
		A := BORDER_A();
		B := BORDER_B();
		R := BORDER_R();
		(totalX, totalY, totalZ) := neighborsRepulsiveBorderEffect(A, B, R, particleID, cellEdgeLength);
	end if;

	x := totalX;
	y := totalY;
	z := totalZ;
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
	input Integer totalNumberOfVolumes;
	input Integer totalNumberOfParticles;
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
	(repulsiveX, repulsiveY, repulsiveZ) := totalRepulsivePedestrianEffect(totalNumberOfParticles, particleID, desiredSpeed, pX, pY, pZ, vX, vY, vZ, targetX, targetY);
	(wallX, wallY, wallZ) := totalRepulsiveBorderEffect(totalNumberOfVolumes, particleID, cellEdgeLength, pX, pY, pZ);

	resultX := accelerationX + repulsiveX + wallX;
	resultY := accelerationY + repulsiveY + wallY;
	resultZ := accelerationZ + repulsiveZ + wallZ;

	x := resultX;
	y := resultY;
	z := resultZ;
end pedestrianTotalMotivation;


end retQSS_social_force_model;