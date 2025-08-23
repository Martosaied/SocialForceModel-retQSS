package retQSS_subway_hub

import retQSS;
import retQSS_utils;
import retQSS_pathways;

function setUpParticleMovement
	input Integer N;
	output Boolean _;
	external "C" _=subway_hub_setUpParticleMovement(N) annotation(
	    Library="subway_hub",
	    Include="#include \"retqss_subway_hub.h\"");
end setUpParticleMovement;

function moveIfNotInSubway
	input Integer particleID;
	input Real hx;
	input Real hy;
	input Real hz;
	output Real dx;
	output Real dy;
	output Real dz;
protected
	Integer currentVolumeID;
	Boolean isSubway;
	Boolean isObjective;
algorithm
	currentVolumeID := particle_currentVolumeID(particleID);
	if currentVolumeID <> 0 then
		isSubway := volume_getProperty(currentVolumeID, "isSubway");
		if isSubway then
			dx := hx;
			dy := hy;
			dz := hz;
		else
			(dx, dy, dz) := getNextPosition(particleID);
		end if;
	else
		dx := hx;
		dy := hy;
		dz := hz;
	end if;
end moveOutOfSubway;

function getInitialStationPosition
	input Integer particleID;
	output Real x;
	output Real y;
	output Real z;
	output Real dx;
	output Real dy;
	output Real dz;
protected
	Real tempX;
	Real tempY;
	Real tempZ;
	Real randomValue;
algorithm
	randomValue := random(0.0, 1.0);
	if randomValue < 0.33 then
		// 1/3 of the particles are in the first subway
		(tempX, tempY, tempZ) := volume_randomPoint(2);
	elseif randomValue < 0.66 then
		// 1/3 of the particles are in the second subway
		(tempX, tempY, tempZ) := volume_randomPoint(10);
	else
		// 1/3 of the particles are in the third subway
		(tempX, tempY, tempZ) := volume_randomPoint(105);
	end if;
	x := tempX;
	y := tempY;
	z := tempZ;
	dx := tempX;
	dy := tempY;
	dz := tempZ;
end getInitialStationPosition;

function getInitialRandomPosition
	input Integer particleID;
	output Real x;
	output Real y;
	output Real z;
	output Real dx;
	output Real dy;
	output Real dz;
protected
	Real randomValue;
	Integer stopID;
	Real fromY;
	Real toY;
	Real fromX;
	Real toX;
	Real tempY;
	Real tempX;
	Real tempZ;
algorithm
	randomValue := random(0.0, 1.0);
	if randomValue < 0.25 then
		// 25% of the particles are in the left side of the grid on the first subway
		fromY := 71;
		toY := 79;
		tempX := random(0, 110);
		x := tempX;
		dx := tempX + 110;
	elseif randomValue < 0.5 then
		// 25% of the particles are in the right side of the grid on the first subway
		fromY := 71;
		toY := 79;
		tempX := random(0, 110);
		x := tempX;
		dx := tempX - 110;
	elseif randomValue < 0.75 then
		// 25% of the particles are in the left side of the grid on the second subway
		fromY := 31;
		toY := 39;
		tempX := random(0, 110);
		x := tempX;
		dx := tempX + 110;
	else
		// 1/6 of the particles are in the left side of the grid on the third subway
		fromY := 31;
		toY := 39;
		tempX := random(0, 110);
		x := tempX;
		dx := tempX - 110;
	end if;
	tempY := random(fromY, toY);
	y := tempY;
	dy := fromY + (toY - fromY) / 2;
	dz := 0.0;
	z := 0.0;
end getInitialRandomPosition;

// function conveyorBeltEffect
// 	input Integer particleID;
// 	input Real GRID_SIZE;
// 	input Real hx;
// 	input Real hy;
// 	input Real hz;
// 	output Real dx;
// 	output Real dy;
// 	output Real dz;
// protected
// 	Real y;
// algorithm
// 	(_, y, _, _, _, _) := getInitialRandomPosition(particleID);
// 	if hx < 1.0 then
// 		dx := GRID_SIZE - 1;
// 		dy := y;
// 		dz := hz;
// 	elseif hx > GRID_SIZE - 1 then
// 		dx := 1.0;
// 		dy := y;
// 		dz := hz;
// 	else
// 		dx := hx;
// 		dy := hy;
// 		dz := hz;
// 	end if;
// end conveyorBeltEffect;

end retQSS_subway_hub;