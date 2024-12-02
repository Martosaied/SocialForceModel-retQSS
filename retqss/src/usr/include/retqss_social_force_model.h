#ifndef _RETQSS_SOCIAL_FORCE_MODEL_H_
#define _RETQSS_SOCIAL_FORCE_MODEL_H_

#include "retqss/retqss_types.hh"

#ifdef __cplusplus

#include <unordered_map>
#include <list>


extern "C"
{
#endif

void social_force_model_totalRepulsivePedestrianEffect(
	int particleID,
	double *desiredSpeed,
	double *pX,
	double *pY,
	double *pZ,
	double *pVX,
	double *pVY,
	double *pVZ,
	double *x,
	double *y,
	double *z);

void social_force_model_totalRepulsiveBorderEffect(
	int particleID,
	double pX[1],
	double pY[1],
	double pZ[1],
	double *x,
	double *y,
	double *z);

void social_force_model_acceleration(
	int particleID,
	double* desiredSpeed,
	double* px,
	double* py,
	double* pz,
	double* vx,
	double* vy,
	double* vz,
	double targetX,
	double targetY,
	double targetZ,
	double *x,
	double *y,
	double *z);

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

#ifdef __cplusplus
}
#endif

#endif
