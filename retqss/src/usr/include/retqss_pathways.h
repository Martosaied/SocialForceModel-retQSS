#ifndef _RETQSS_PATHWAYS_H_
#define _RETQSS_PATHWAYS_H_

#include "retqss/retqss_types.hh"

#ifdef __cplusplus

#include <vector>
#include <deque>

extern "C"
{
#endif

Bool pathways_setUpPathways();

Bool pathways_setUpRandomPathways(int particleID, int size);

int pathways_getRandomPathway();

int pathways_getCurrentStop(int particleID);

int pathways_getNextStop(int particleID);

void pathways_nextStop(
	int particleID,
	double currentDx,
	double currentDy,
	double currentDz,
	double *dx,
	double *dy,
	double *dz);
	
#ifdef __cplusplus

Bool pathways_addPathway(std::deque<int> pathway);
Bool pathways_setPathway(int particleID, std::deque<int> pathway);

}
#endif

#endif