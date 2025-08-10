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

std::vector<std::deque<int>> pathways;
std::map<int, std::deque<int>> particle_pathways;

extern "C"
{

std::deque<int> pathways_generateRandomPathway(int pathSize) {
	std::deque<int> pathway;
	for (int i = 1; i <= pathSize; i++) {
		pathway.push_back(rand() % retQSS_geometry_countVolumes() + 1);
	}

	return pathway;
}
	

Bool pathways_setUpPathways() {
	// Pathways setup logic can be implemented here
	// This function can initialize specific pathways properties
	// such as platform areas, entrance/exit points, etc.
	return true;
}

Bool pathways_setUpRandomPathways(int particleID, int size) {
	std::deque<int> pathway = pathways_generateRandomPathway(size);
	particle_pathways[particleID] = pathway;
	
	return true;
}

int pathways_getRandomPathway() {
	return rand() % pathways.size();
}

int pathways_getCurrentStop(int particleID) {
	return particle_pathways[particleID].front();
}

int pathways_getNextStop(int particleID) {
	// If the particleID is 0, return 0
	if (particleID == 0) {
		return 0;
	}

	// If the particle has no pathway, return 0
	if (particle_pathways[particleID].size() == 0) {
		return 0;
	}

	// If the particle has a pathway, return the next stop
	int nextStop = particle_pathways[particleID].front();
	int currentVolume = retQSS_particle_currentVolumeID(particleID);

	// If the particle is not at the next stop, return the next stop
	if (currentVolume != nextStop) {
		return nextStop;
	}

	// If the particle is at the next stop, remove the first stop from the pathway and return the following stop
	if (particle_pathways[particleID].size() > 1) {
		particle_pathways[particleID].pop_front();
	}
	return particle_pathways[particleID].front();
}

Bool pathways_addPathway(std::deque<int> pathway) {
	pathways.push_back(pathway);
	return true;
}

Bool pathways_setPathway(int particleID, std::deque<int> pathway) {
	particle_pathways[particleID] = pathway;
	return true;
}
}