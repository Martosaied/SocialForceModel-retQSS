#ifndef _RETQSS_SUBWAY_STATIONS_H_
#define _RETQSS_SUBWAY_STATIONS_H_

#include "retqss/retqss_types.hh"

#ifdef __cplusplus

#include <unordered_map>
#include <list>
#include <vector>

extern "C"
{
#endif

void classrooms_initContiguousHallways(int gridDivisions);

void classrooms_nearestHallwayPosition(
	int particleID,
	double currentDx,
	double currentDy,
	double currentDz,
	double *dx,
	double *dy,
	double *dz);

void classrooms_randomConnectedHallway(
	int particleID,
	double currentDx,
	double currentDy,
	double currentDz,
	double *dx,
	double *dy,
	double *dz);

void classrooms_randomInitialClassroomPosition(
	int particleID,
	double *x,
	double *y,
	double *z,
	double *dx,
	double *dy,
	double *dz);

#ifdef __cplusplus
}
#endif

#endif