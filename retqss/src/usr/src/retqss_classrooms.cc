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

extern "C"
{

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

	int nextHallway = hallways[rand() % hallways.size()];
	retQSS_volume_randomPoint(nextHallway, dx, dy, dz);
}

void classrooms_randomInitialClassroomPosition(int particleID, double *x, double *y, double *z, double *dx, double *dy, double *dz) {
	std::vector<int> classrooms = utils_getArrayParameter("CLASSROOMS");
	int nextClassroom = classrooms[rand() % classrooms.size()];
	retQSS_volume_randomPoint(nextClassroom, x, y, z);
	retQSS_particle_setProperty(particleID, "classroomID", nextClassroom);

	*dx = *x;
	*dy = *y;
	*dz = *z;
}

}