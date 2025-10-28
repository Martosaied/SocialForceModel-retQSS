#include "retqss_classrooms.h"

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
#include <random>

extern "C"
{

std::vector<std::set<int>> contiguousHallways;

int getRandomElementFromSet(const std::set<int>& s) {
    if (s.empty()) return -1; // or throw an exception

    // Random device & generator
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<> dist(0, s.size() - 1);

    int randomIndex = dist(gen);
    auto it = s.begin();
    std::advance(it, randomIndex);

    return *it;
}

bool isValidIndex(int id, int gridDivisions) {
    return id >= 1 && id <= gridDivisions * gridDivisions;
}

void dfsHallways(int id, int gridDivisions, std::set<int>& visited, std::set<int>& component) {
    if (visited.count(id)) return;
	if (!isValidIndex(id, gridDivisions)) return;
    if (!retQSS_volume_getProperty(id, "isHallway")) return;

    visited.insert(id);
    component.insert(id);

    std::vector<int> neighbors;
    neighbors.push_back(id - gridDivisions); // left
    neighbors.push_back(id + gridDivisions); // right
    neighbors.push_back(id - 1);             // down
    neighbors.push_back(id + 1);             // up

    for (int n : neighbors)
        dfsHallways(n, gridDivisions, visited, component);
}

std::vector<std::set<int>> classrooms_getAllContiguousHallwayGroups(int gridDivisions) {
    std::set<int> visited;
    std::vector<std::set<int>> allGroups;

    int totalVolumes = gridDivisions * gridDivisions;

    for (int id = 1; id <= totalVolumes; ++id) {
        if (!visited.count(id) && retQSS_volume_getProperty(id, "isHallway")) {
            std::set<int> component;
            dfsHallways(id, gridDivisions, visited, component);
            allGroups.push_back(component);
        }
    }

    return allGroups;
}

void classrooms_initContiguousHallways(int gridDivisions) {
	contiguousHallways = classrooms_getAllContiguousHallwayGroups(gridDivisions);
}


void classrooms_nearestHallwayPosition(int particleID, double currentDx, double currentDy, double currentDz, double *dx, double *dy, double *dz) {
	int currentVolumeID = retQSS_particle_currentVolumeID(particleID);
	if (currentVolumeID == 0 || !retQSS_volume_getProperty(currentVolumeID, "isClassroom")) {
		*dx = currentDx;
		*dy = currentDy;
		*dz = currentDz;
		return;
	}

	double currentX, currentY, currentZ;
	retQSS_particle_currentPosition(particleID, &currentX, &currentY, &currentZ);

	double closestDistance = std::numeric_limits<double>::max();
	std::vector<int> hallways = utils_getArrayParameter("HALLWAYS");
	for (int i : hallways) {
		double distance = retQSS_volume_distanceToPoint(i, currentX, currentY, currentZ);
		if (distance < closestDistance) {
			closestDistance = distance;
			retQSS_volume_randomPoint(i, dx, dy, dz);
		}
	}
}

void classrooms_randomConnectedHallway(int particleID, double currentDx, double currentDy, double currentDz, double *dx, double *dy, double *dz) {
	std::vector<int> hallways = utils_getArrayParameter("HALLWAYS");

	if (hallways.size() == 0) {
		*dx = currentDx;
		*dy = currentDy;
		*dz = currentDz;
		return;
	}

	int currentVolumeID = retQSS_particle_currentVolumeID(particleID);
	for (std::set<int> contiguousHallway : contiguousHallways) {
		if (contiguousHallway.count(currentVolumeID)) {
			int randomHallway = getRandomElementFromSet(contiguousHallway);
			retQSS_volume_randomPoint(randomHallway, dx, dy, dz);
			return;
		}
	}
}

void classrooms_randomInitialClassroomPosition(int particleID, double *x, double *y, double *z, double *dx, double *dy, double *dz) {
	std::vector<int> classrooms = utils_getArrayParameter("CLASSROOMS");
	int nextClassroom = classrooms[rand() % classrooms.size()];
	retQSS_volume_centroid(nextClassroom, x, y, z);
	double randomX = -2.0 + (std::rand() / (double)RAND_MAX) * 4.0;
	double randomY = -2.0 + (std::rand() / (double)RAND_MAX) * 4.0;
	*x += randomX;
	*y += randomY;
	retQSS_particle_setProperty(particleID, "classroomID", nextClassroom);

	*dx = *x;
	*dy = *y;
	*dz = *z;
}

}