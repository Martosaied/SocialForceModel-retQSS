package retQSS_pathways

import retQSS;
import retQSS_utils;
import retQSS_social_force_model_types;

function setUpPathways
	output Boolean _;
	external "C" _=pathways_setUpPathways() annotation(
	    Library="pathways",
	    Include="#include \"retqss_pathways.h\"");
end setUpPathways;

function setUpRandomPathways
	input Integer particleID;
	input Integer size;
	output Boolean _;
	external "C" _=pathways_setUpRandomPathways(particleID, size) annotation(
	    Library="pathways",
	    Include="#include \"retqss_pathways.h\"");
end setUpRandomPathways;

function getRandomPathway
	output Integer pathwayID;
	external "C" pathwayID=pathways_getRandomPathway() annotation(
	    Library="pathways",
	    Include="#include \"retqss_pathways.h\"");
end getRandomPathway;

function getCurrentStop
	input Integer particleID;
	output Integer stopID;
	external "C" stopID=pathways_getCurrentStop(particleID) annotation(
	    Library="pathways",
	    Include="#include \"retqss_pathways.h\"");
end getCurrentStop;

function getNextStop
	input Integer particleID;
	output Integer stopID;
	external "C" stopID=pathways_getNextStop(particleID) annotation(
	    Library="pathways",
	    Include="#include \"retqss_pathways.h\"");
end getNextStop;

function getNextPosition
	input Integer particleID;
	output Real x;
	output Real y;
	output Real z;
protected
	Integer stopID;
algorithm
	stopID := getNextStop(particleID);
	(x, y, z) := volume_randomPoint(stopID);
end getNextPosition;

function getInitialPosition
	input Integer particleID;
	output Real x;
	output Real y;
	output Real z;
	output Real dx;
	output Real dy;
	output Real dz;
protected
	Integer stopID;
	Real rx;
	Real ry;
	Real rz;
algorithm
	stopID := getCurrentStop(particleID);
	(rx, ry, rz) := volume_randomPoint(stopID);
	dx := rx;
	dy := ry;
	dz := rz;
	x := rx;
	y := ry;
	z := rz;
end getInitialPosition;

end retQSS_pathways;