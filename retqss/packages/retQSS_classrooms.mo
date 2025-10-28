package retQSS_classrooms

import retQSS;
import retQSS_utils;
import retQSS_social_force_model_types;

function initContiguousHallways
	input Integer gridDivisions;
	output Boolean _;
external "C" classrooms_initContiguousHallways(gridDivisions) annotation(
	    Library="classrooms",
	    Include="#include \"retqss_classrooms.h\"");
end initContiguousHallways;

function randomInitialClassroomPosition
	input Integer particleID;
	output Real x;
	output Real y;
	output Real z;
	output Real dx;
	output Real dy;
	output Real dz;
external "C" classrooms_randomInitialClassroomPosition(particleID, x, y, z, dx, dy, dz) annotation(
	    Library="classrooms",
	    Include="#include \"retqss_classrooms.h\"");
end randomInitialClassroomPosition;

function nearestHallwayPosition
	input Integer particleID;
	input Real currentDx;
	input Real currentDy;
	input Real currentDz;
	output Real dx;
	output Real dy;
	output Real dz;
	external "C" classrooms_nearestHallwayPosition(particleID, currentDx, currentDy, currentDz, dx, dy, dz) annotation(
	    Library="classrooms",
	    Include="#include \"retqss_classrooms.h\"");
end nearestHallwayPosition;

function randomConnectedHallway
	input Integer particleID;
	input Real currentDx;
	input Real currentDy;
	input Real currentDz;
	output Real dx;
	output Real dy;
	output Real dz;
	external "C" classrooms_randomConnectedHallway(particleID, currentDx, currentDy, currentDz, dx, dy, dz) annotation(
	    Library="classrooms",
	    Include="#include \"retqss_classrooms.h\"");
end randomConnectedHallway;

end retQSS_subway_stations;