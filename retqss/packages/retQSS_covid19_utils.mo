package retQSS_covid19_utils

import retQSS;
import retQSS_utils;

/*
  Returns a random 3d-point given an XY square size and a fixed z-coordinate
*/
function randomXYPoint
	input Real size;
	input Real zCoord;
	output Real x;
	output Real y;
	output Real z;
algorithm
	x := random(0.0, size);
	y := random(0.0, size);
	z := zCoord;
end randomXYPoint;

/*
  Returns a random boolean given the probability of being true
*/
function randomBoolean
	input Real trueProbability;
	output Boolean result;
algorithm
	if random(0.0, 1.0) < trueProbability then
		result := true;
	else
		result := false;
	end if;
end randomBoolean;

/*
  Returns a random point in a random classroom.
  The function is used in the context of the heterogeneous school scenario
*/
function randomXYClassroom
	input Real cellSize;
	input Real zCoord;
	output Real x;
	output Real y;
	output Real z;
protected
	Real rx;
	Real ry;
algorithm
	(rx, ry, z) := randomXYPoint(cellSize, zCoord);
	if randomBoolean(0.5) then
		x := rx + cellSize*2;
	else
		x := rx + cellSize*4;
	end if;
	if randomBoolean(0.5) then
		ry := ry + cellSize*4;
	end if;
	if randomBoolean(0.5) then
		ry := ry + cellSize*2;
	end if;
	y := ry;
end randomXYClassroom;

/*
  Returns if a given volume id is a hallway
  The function is used in the context of the heterogeneous school scenario
*/
function isSchoolHallway
	input Integer volumeID;
	output Boolean hallway;
algorithm
	if (22 <= volumeID and volumeID <= 28) then
		hallway := true;
	else
		hallway := false;
	end if;
end isSchoolHallway;

/*
  Sets the "isBlock" property on a volume. That property determines if that given volume is transitable or not by a particle.
  This is used to make the map of the classroom and also in the logic of letting the students go out of the classroom to the hallway and then return.
  The function is used in the context of the heterogeneous school scenario
*/
function setClassroomBlock
	input Integer volumeID;
	input Integer inClassTime;
	output Boolean _;
protected
	Real blocked;
algorithm
	if (22 <= volumeID and volumeID <= 28) then
		blocked := inClassTime;
	elseif (volumeID == 15 or volumeID == 17) or (volumeID == 19 or volumeID == 21) or (volumeID == 29 or volumeID == 31) or (volumeID == 33 or volumeID == 35) then
		blocked := 1 - inClassTime;
	else
		blocked := 1;
	end if;
	_ := volume_setProperty(volumeID, "isBlock", blocked);
end setClassroomBlock;

/*
  Gets the (x,y) point of the classroom door for the current volume a given particle.
  The function is used in the context of the heterogeneous school scenario
*/
function getClassroomDoor
	input Integer particleID;
	input Real edge;
	output Real x;
	output Real y;
protected
	Real dx;
algorithm
	if particle_getProperty(particleID, "initialX") < edge * 3.5 then
		dx := 1.1;
	else
		dx := -0.1;
	end if;
	x := (floor(particle_getProperty(particleID, "initialX")/edge)+dx)*edge;
	y := (floor(particle_getProperty(particleID, "initialY")/edge)+0.5)*edge;
end getClassroomDoor;

/*
  Returns a random (x,y) vector given its distance from origen (r)
*/
function randomXYVector
	input Real r;
	output Real x;
	output Real y;
protected
	Real alfa;
algorithm
	alfa := random(0.0, 2.0 * PI());
	x := r*cos(alfa);
	y := r*sin(alfa);
end randomXYVelocity;

/*
  Returns a random 3d vector to be used as the velocity of a particle
  Z coordinate will be 0
*/
function randomXYVelocity
	input Real defaultSpeed;
	input Real superSpreaderProb;
	input Real superSpreaderAcceleration;
	input Real noVelocityProb;
	output Real x;
	output Real y;
	output Real z;
protected
	Real ps;
	Real speed;
algorithm
    ps := random(0., 1.);
	if ps < superSpreaderProb then
	    speed := defaultSpeed * superSpreaderAcceleration;
	elseif ps < superSpreaderProb + noVelocityProb then 
	    speed := 0.0;
	else
	    speed := defaultSpeed;
	end if;
	(x, y) := randomXYVector(speed);
	z := 0;
end randomXYVelocity;

/*
  Returns a random 3d vector to be used as the velocity using the volume properties of a given particle
  Z coordinate will be 0
*/
function randomXYVolumeVelocity
	input Integer particleID;
	output Real x;
	output Real y;
protected
	Integer volumeID;
	Real speed;
	Boolean _;
algorithm
	volumeID := particle_currentVolumeID(particleID);
	speed := volume_getProperty(volumeID, "particleSpeed");
	(x, y) := randomXYVector(speed);
end randomXYVolumeVelocity;

/*
  Computes the natural/newtonian bounce velocity vector of a particle against a border given the particle velocity and the border's normal vector
*/
function naturalBounceVelocity
	input Real normalX;
	input Real normalY;
	input Real currentVx;
	input Real currentVy;
	output Real ux;
	output Real uy;
algorithm
	if abs(normalX) < EPS() then
		ux := currentVx;
		uy := -currentVy;
	else
		ux := -currentVx;
		uy := currentVy;
	end if;
end naturalBounceVelocity;

/*
  Computes a random bounce velocity vector of a particle against a border given the particle velocity and the border's normal vector
*/
function randomBounceVelocity
	input Real normalX;
	input Real normalY;
	input Real currentVx;
	input Real currentVy;
	output Real ux;
	output Real uy;
protected
	Real speed;
	Real alfa;
	Real normalAlfa;
	Real deltaAlfa;
algorithm
	if abs(normalX) < EPS() then
		alfa := atan2(-currentVy, currentVx);
	else
		alfa := atan2(currentVy, -currentVx);
	end if;
	normalAlfa := atan2(-normalY, -normalX);
	deltaAlfa := alfa - normalAlfa;
	if deltaAlfa < -PI() then
		deltaAlfa := deltaAlfa + 2.0 * PI();
	elseif deltaAlfa > PI() then
		deltaAlfa := deltaAlfa - 2.0 * PI();
	end if;
	if abs(deltaAlfa) < PI() / 4.0 then
		alfa := random(normalAlfa - PI() / 4.0, normalAlfa + PI() / 4.0);
	end if;
	speed := sqrt(currentVx*currentVx+currentVy*currentVy);
	ux := speed * cos(alfa);
	uy := speed * sin(alfa);
end randomBounceVelocity;

/*
  Increments by 1 the value of a given particle property
*/
function incrementProperty
	input Integer particleID;
	input String property;
	output Boolean _;
protected
	Integer v;
algorithm
	v := particle_getProperty(particleID, property);
	_ := particle_setProperty(particleID, property, v + 1.0);
end incrementProperty;

/*
  Dump the state of the whole model (states of all particles and volumes) in a CSV line
  This function is implemented in C
*/
function covid19_outputCSV
	input Real time;
	input Integer N;
	input Real x[1];
	input Real y[1];
	input Integer V;
	input Real volumeConcentration[1];
	input Integer recoveredCount;
	input Integer infectionsCount;
	output Boolean status;
	external "C" status=covid19_outputCSV(time, N, x, V, volumeConcentration, recoveredCount, infectionsCount) annotation(
	    Library="covid19",
	    Include="#include \"retqss_covid19.h\"");
end covid19_outputCSV;

end retQSS_covid19_utils;
