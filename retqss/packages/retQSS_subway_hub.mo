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

function moveOutOfSubway
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
	isSubway := volume_getProperty(currentVolumeID, "isSubway");
	isObjective := particle_getProperty(particleID, "isObjective");

	if isSubway and isObjective then
		dx := hx;
		dy := hy;
		dz := hz;
	else
		(dx, dy, dz) := getNextPosition(particleID);
	end if;
end moveOutOfSubway;

end retQSS_subway_hub;