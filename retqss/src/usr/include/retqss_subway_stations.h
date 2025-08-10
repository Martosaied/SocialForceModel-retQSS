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

Bool subway_stations_setUpStations();

Bool subway_stations_setUpPedestrianVolumePaths(int particleID, int pathSize);

void subway_stations_nextStation(
	int particleID,
	double currentDx,
	double currentDy,
	double currentDz,
	double *dx,
	double *dy,
	double *dz);

int subway_stations_getRandomConnectedStation(int volumeID);

int subway_stations_calculateRandomConnectedStation(int particleID);

void subway_stations_randomConnectedStation(
	int particleID,
	double currentDx,
	double currentDy,
	double currentDz,
	double *dx,
	double *dy,
	double *dz);

#ifdef __cplusplus
}
#endif

#endif