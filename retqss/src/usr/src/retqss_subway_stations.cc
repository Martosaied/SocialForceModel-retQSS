#include "retqss_subway_stations.h"

#include "retqss/retqss_model_api.h"
#include "retqss/retqss_interface.hh"
#include "retqss/retqss_utilities.hh"
#include "retqss_utils.hh"

#include <cmath>
#include <fstream>
#include <set>
#include <chrono>
#include <ctime>
#include <cstddef>
#include <algorithm>

std::map<int, std::deque<int>> pedestrian_volume_paths;

extern "C"
{

Bool subway_stations_setUpStations() {
	// Station setup logic can be implemented here
	// This function can initialize specific subway station properties
	// such as platform areas, entrance/exit points, etc.
	return true;
}

int subway_stations_getRandomConnectedStation(int volumeID) {
	std::list<int> stations;
	int volumes = retQSS_geometry_countVolumes();
	for (int i = 1; i <= volumes; i++) {
		if (retQSS_volume_getProperty(i, "isStation")) {
			stations.push_back(i);
		}
	}

	if (stations.size() == 0) {
		return 0;
	}

	std::vector<int> neighboringStations;
	for (int station : stations) {
		int idDifference = abs(station - volumeID);
		// Subway line connectivity logic - stations are connected if they differ by 55 or 5
		// This assumes a specific grid layout for the subway system
		if (idDifference == 55 || idDifference == 5) {
			neighboringStations.push_back(station);
		}
	}

	if (neighboringStations.size() == 0) {
		// If no neighboring stations found, return a random station
		int index = rand() % stations.size();
		auto it = stations.begin();
		std::advance(it, index);
		return *it;
	}

	// Get a random neighboring station
	int index = rand() % neighboringStations.size();
	int nextStation = neighboringStations[index];

	return nextStation;
}

int subway_stations_calculateRandomConnectedStation(int particleID) {
	int currentVolumeID = retQSS_particle_currentVolumeID(particleID);
	if (currentVolumeID == 0 || !retQSS_volume_getProperty(currentVolumeID, "isStation")) {
		return currentVolumeID;
	}

	return subway_stations_getRandomConnectedStation(currentVolumeID);
}

void subway_stations_randomConnectedStation(
	int particleID,
	double currentDx,
	double currentDy,
	double currentDz,
	double *dx,
	double *dy,
	double *dz)
{
	int nextStation = subway_stations_calculateRandomConnectedStation(particleID);

	// Get a random point in the station
	double centroidX, centroidY, centroidZ;
	retQSS_volume_centroid(nextStation, &centroidX, &centroidY, &centroidZ);
	*dx = centroidX;
	*dy = centroidY;
	*dz = centroidZ;
}

void subway_stations_randomNextStation(
	int particleID,
	double currentDx,
	double currentDy,
	double currentDz,
	double *dx,
	double *dy,
	double *dz)
{
	subway_stations_randomConnectedStation(particleID, currentDx, currentDy, currentDz, dx, dy, dz);
}

Bool subway_stations_setUpPedestrianVolumePaths(int particleID, int pathSize) {
	int currentVolumeID = retQSS_particle_currentVolumeID(particleID);
	pedestrian_volume_paths[particleID] = std::deque<int>();
	
	for (int j = 0; j < pathSize; j++) {
		int nextStation = subway_stations_getRandomConnectedStation(currentVolumeID);
		
		// Avoid revisiting the same station
		while (std::find(pedestrian_volume_paths[particleID].begin(), 
		                 pedestrian_volume_paths[particleID].end(), 
		                 nextStation) != pedestrian_volume_paths[particleID].end()) {
			nextStation = subway_stations_getRandomConnectedStation(currentVolumeID);
		}
		
		pedestrian_volume_paths[particleID].push_back(nextStation);
		currentVolumeID = nextStation;
	}

	return true;
}

void subway_stations_nextStation(
	int particleID,
	double currentDx,
	double currentDy,
	double currentDz,
	double *dx,
	double *dy,
	double *dz)
{
	int currentStation = retQSS_particle_currentVolumeID(particleID);
	if (pedestrian_volume_paths[particleID].size() == 0) {
		// return the current position
		*dx = currentDx;
		*dy = currentDy;
		*dz = currentDz;
		return;
	}

	// Pop the next station from the path
	int nextStation = pedestrian_volume_paths[particleID].front();
	pedestrian_volume_paths[particleID].pop_front();

	// Get a random point in the station
	double randomX, randomY, randomZ;
	retQSS_volume_randomPoint(nextStation, &randomX, &randomY, &randomZ);
	*dx = randomX;
	*dy = randomY;
	*dz = randomZ;
}

}