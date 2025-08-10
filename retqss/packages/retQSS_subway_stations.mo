package retQSS_subway_stations

import retQSS;
import retQSS_utils;
import retQSS_social_force_model_types;

function setUpStations
	output Boolean _;
	external "C" _=subway_stations_setUpStations() annotation(
	    Library="subway_stations",
	    Include="#include \"retqss_subway_stations.h\"");
end setUpStations;

function randomInitialSubwayPosition
	output Real x;
	output Real y;
	output Real z;
	output Real dx;
	output Real dy;
	output Real dz;
protected
	Integer i0;
	Integer randomValue;
	Integer rX;
	Integer rY;
	Integer rZ;
algorithm
	for i0 in 1:10000 loop
		randomValue := geometry_randomVolumeID();
		if volume_getProperty(randomValue, "isStation") then
			(rX, rY, rZ) := volume_randomPoint(randomValue);
			x := rX;
			y := rY;
			z := rZ;
			dx := rX;
			dy := rY;
			dz := rZ;
			return;
		end if;
	end for;
end randomInitialSubwayPosition;

function nextStation
	input Integer particleID;
	input Real currentDx;
	input Real currentDy;
	input Real currentDz;
	output Real dx;
	output Real dy;
	output Real dz;
	external "C" subway_stations_nextStation(particleID, currentDx, currentDy, currentDz, dx, dy, dz) annotation(
	    Library="subway_stations",
	    Include="#include \"retqss_subway_stations.h\"");
end nextStation;

function randomNextStation
	input Integer particleID;
	input Real currentDx;
	input Real currentDy;
	input Real currentDz;
	output Real dx;
	output Real dy;
	output Real dz;
	external "C" subway_stations_randomNextStation(particleID, currentDx, currentDy, currentDz, dx, dy, dz) annotation(
	    Library="subway_stations",
	    Include="#include \"retqss_subway_stations.h\"");
end randomNextStation;

function updateInStationPosition
	input Integer particleID;
	output Real dx;
	output Real dy;
	output Real dz;
protected
	Real x;
	Real y;
	Real z;
	Integer nextVolume;
	Boolean isStation;
algorithm
	nextVolume := particle_nextVolumeID(particleID);
	if nextVolume <> 0 then
		isStation := volume_getProperty(nextVolume, "isStation");
		if isStation then
			(x, y, z) := volume_randomPoint(nextVolume);
			dx := x;
			dy := y;
			dz := z;
			return;
		end if;
		(dx, dy, dz) := volume_centroid(nextVolume);
	end if;
end updateInStationPosition;

function setUpPedestrianVolumePaths
	input Integer particleID;
	input Integer pathSize;
	output Boolean _;
	external "C" _=subway_stations_setUpPedestrianVolumePaths(particleID, pathSize) annotation(
	    Library="subway_stations",
	    Include="#include \"retqss_subway_stations.h\"");
end setUpPedestrianVolumePaths;

end retQSS_subway_stations;