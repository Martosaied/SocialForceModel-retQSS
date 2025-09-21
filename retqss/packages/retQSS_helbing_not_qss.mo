package retQSS_helbing_not_qss

import retQSS;
import retQSS_utils;
import retQSS_social_force_model_params;
import retQSS_social_force_model_utils;
import retQSS_social_force_model_types;

function setParameters
	output Boolean _;
	external "C" _=social_force_model_setParameters() annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end setParameters;

function outputCSV
	input Real time;
	input Integer N;
	input Real x[1];
	input Real y[1];
	input Real vx[1];
	input Real vy[1];
	input Integer groupIDs[1];
	output Boolean status;
	external "C" status=social_force_model_notQSS_outputCSV(time, N, x, groupIDs) annotation(
	    Library="social_force_model",
	    Include="#include \"retqss_social_force_model.h\"");
end outputCSV;

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
		randomValue2 := random(0.1, size/3);
		// randomValue2 := 0.1 * size;
		// randomValue2 := random(0, size);
		x := randomValue2;
		dx := 1.5 * size;
		group := left;
	else
		randomValue2 := random(size/3 * 2, size);
		// randomValue2 := 0.9 * size;
		// randomValue2 := random(0, size);
		x := randomValue2;
		dx := -0.5 * size;
		group := right;
	end if;
	randomValue := random(fromY, toY);
	dy := randomValue;
	y := randomValue;
	dz := 0.0;
	z := 0.0;
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
algorithm
	(desiredX, desiredY, desiredZ) := vectorWithNorm((targetX - currentX), (targetY - currentY), (targetZ - currentZ), 1.0);
end desiredDirection;


function acceleration
	input Integer particleID;
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
    desiredSpeedValue := 1.34;

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

	relaxationTime := 1/0.5; // 0.5 seconds
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
	input Real targetX;
	input Real targetY;
	input Real targetZ;
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
                targetX, targetY, targetZ
            );
            totalRepulsiveX := totalRepulsiveX + repulsiveX;
            totalRepulsiveY := totalRepulsiveY + repulsiveY;
            totalRepulsiveZ := totalRepulsiveZ + repulsiveZ;
        end if;
    end for;

	x := totalRepulsiveX;
	y := totalRepulsiveY;
	z := totalRepulsiveZ;	
end totalRepulsivePedestrianEffect;

function repulsiveBorderEffect
	input Integer particleID;
	input Real X[1];
	input Real Y[1];
	input Real Z[1];
	input Real VX[1];
	input Real VY[1];
	input Real VZ[1];
	input Real corridorWidth;
	input Real gridSize;
	output Real x;
	output Real y;
	output Real z;
protected
	Real A;
	Real B;
	Real R;
	Real pX;
	Real pY;
	Real pZ;
	Real vX;
	Real vY;
	Real vZ;
	Real corridorTopY;
	Real corridorBottomY;
	Real distanceToTop;
	Real distanceToBottom;
	Real nearestWallY;
	Real distanceToWall;
	Real normalizedY;
	Real fy;
algorithm
	A := BORDER_A();
	B := BORDER_B();
	R := BORDER_R();

	pX := equationArrayGet(X, particleID);
	pY := equationArrayGet(Y, particleID);
	pZ := equationArrayGet(Z, particleID);
	vX := equationArrayGet(VX, particleID);
	vY := equationArrayGet(VY, particleID);
	vZ := equationArrayGet(VZ, particleID);

	// Calculate the Y positions of the two horizontal corridor walls
	corridorTopY := (gridSize / 2.0) + (corridorWidth / 2.0);
	corridorBottomY := (gridSize / 2.0) - (corridorWidth / 2.0);

	// Calculate distances to both walls
	distanceToTop := abs(pY - corridorTopY);
	distanceToBottom := abs(pY - corridorBottomY);

	// Find the nearest wall
	nearestWallY := 0.0;
	distanceToWall := 0.0;
	if distanceToTop < distanceToBottom then
		nearestWallY := corridorTopY;
		distanceToWall := distanceToTop;
	else
		nearestWallY := corridorBottomY;
		distanceToWall := distanceToBottom;
	end if;

	// Calculate normalized direction to the nearest wall (only Y component)
	normalizedY := (pY - nearestWallY) / distanceToWall;
	
	// Apply repulsive force only in Y direction
	fy := A*exp((R-distanceToWall)/B)*normalizedY;

	x := 0;  // No force in X direction
	y := fy; // Only force in Y direction
	z := 0;  // No force in Z direction
end repulsiveBorderEffect;

function totalRepulsiveBorderEffect
	input Integer particleID;
	input Real X[1];
	input Real Y[1];
	input Real Z[1];
	input Real VX[1];
	input Real VY[1];
	input Real VZ[1];
	output Real x;
	output Real y;
	output Real z;
protected
	Real repulsiveX;
	Real repulsiveY;
	Real repulsiveZ;
	Real gridSize;
	Real corridorWidth;
	Real fromY;
	Real toY;
algorithm
	// Get parameters from the model
	gridSize := GRID_SIZE();
	fromY := FROM_Y();
	toY := TO_Y();
	
	// Calculate corridor width from FROM_Y and TO_Y parameters
	corridorWidth := toY - fromY;

	(repulsiveX, repulsiveY, repulsiveZ) := repulsiveBorderEffect(particleID, X, Y, Z, VX, VY, VZ, corridorWidth, gridSize);

	x := repulsiveX;
	y := repulsiveY;
	z := repulsiveZ;
end totalRepulsiveBorderEffect;


function pedestrianTotalMotivation
	input Integer particleID;
	input Integer totalNumberOfParticles;
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
	Real currentVX;
	Real currentVY;
	Real currentVZ;
	Real repulsiveX;
	Real repulsiveY;
	Real repulsiveZ;
	Real wallX;
	Real wallY;
	Real wallZ;
	Real accelerationX;
	Real accelerationY;
	Real accelerationZ;
	Real resultX;
	Real resultY;
	Real resultZ;
algorithm
	(accelerationX, accelerationY, accelerationZ) := acceleration(particleID, pX, pY, pZ, vX, vY, vZ, targetX, targetY, targetZ);
	(repulsiveX, repulsiveY, repulsiveZ) := totalRepulsivePedestrianEffect(totalNumberOfParticles, particleID, pX, pY, pZ, vX, vY, vZ, targetX, targetY, targetZ);
	(wallX, wallY, wallZ) := totalRepulsiveBorderEffect(particleID, pX, pY, pZ, vX, vY, vZ);

	resultX := accelerationX + repulsiveX + wallX;
	resultY := accelerationY + repulsiveY + wallY;
	resultZ := accelerationZ + repulsiveZ + wallZ;

	x := resultX;
	y := resultY;
	z := resultZ;
end pedestrianTotalMotivation;


end retQSS_helbing_not_qss;