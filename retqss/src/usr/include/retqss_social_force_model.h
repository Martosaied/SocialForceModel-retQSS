#ifndef _RETQSS_SOCIAL_FORCE_MODEL_H_
#define _RETQSS_SOCIAL_FORCE_MODEL_H_

#include "retqss/retqss_types.hh"

#ifdef __cplusplus

#include <unordered_map>
#include <list>


extern "C"
{
#endif


void social_force_model_desiredDirection(
	double currentX,
	double currentY,
	double currentZ,
	double targetX,
	double targetY,
	double targetZ,
	double *desiredX,
	double *desiredY,
	double *desiredZ);

void social_force_model_pedestrianAcceleration(
	double pX,
	double pY,
	double pZ,
	double vX,
	double vY,
	double vz,
	double targetX,
	double targetY,
	double targetZ,
	double *x,
	double *y,
	double *z
);

void social_force_model_repulsivePedestrianEffect(
	double pX1, double pY1, double pZ1,
	double pX2, double pY2, double pZ2,
	double vX2, double vY2, double vZ2,
	double *x, double *y, double *z
);

#ifdef __cplusplus
}
#endif

#endif
