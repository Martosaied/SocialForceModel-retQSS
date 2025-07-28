#ifndef _RETQSS_SOCIAL_FORCE_MODEL_H_
#define _RETQSS_SOCIAL_FORCE_MODEL_H_

#include "retqss/retqss_types.hh"

#ifdef __cplusplus

#include <unordered_map>
#include <list>

#include <vector>
#include "retqss/retqss_cgal_main_types.hh"
#include "retqss/retqss_particle_neighbor.hh"
#include "retqss/retqss_particle_neighborhood.hh"

extern "C"
{
#endif

struct Wall
{
	double from_x;
	double from_y;
	double to_x;
	double to_y;
};

struct ModelParameters {
	double PEDESTRIAN_A_1;
	double PEDESTRIAN_B_1;
	double PEDESTRIAN_A_2;
	double PEDESTRIAN_B_2;
	double PEDESTRIAN_R;
	double PEDESTRIAN_LAMBDA;
	double PEDESTRIAN_IMPLEMENTATION;
	double BORDER_IMPLEMENTATION;
	double BORDER_A;
	double BORDER_B;
	double BORDER_R;
	double FORCE_TERMINATION_AT;
	double GRID_SIZE;
	double FROM_Y;
	double TO_Y;
};

Bool social_force_model_setUpParticles(
    int N,
    double cellEdgeLength,
    int gridDivisions,
    double *x);

Bool social_force_model_setUpWalls();

Bool social_force_model_setParameters();

Bool social_force_model_outputCSV(double time, 
	int N,
	double *x);

void social_force_model_updateNeighboringVolumes(int particleID, int gridDivisions);

void social_force_model_neighborsRepulsiveBorderEffect(
	double A,
	double B,
	double r,
	int particleID,
	double cellEdgeLength,
	double *x,
	double *y, 
	double *z
);

void social_force_model_volumeBasedRepulsivePedestrianEffect(
	int particleID,
	double targetX,
	double targetY,
	double *totalRepulsiveX,
	double *totalRepulsiveY,
	double *totalRepulsiveZ
);

// void repulsive_pedestrian_effect(
// 	retQSS::ParticleNeighbor *neighbors,
// 	const std::vector<double> &args,
// 	Vector_3 &result);

void social_force_model_repulsiveBorderEffect(
	double A,
	double B,
	double ra,
	int particleID,
	double *x,
	double *y,
	double *z);

void social_force_model_totalRepulsivePedestrianEffect(
	int particleID,
	double *desiredSpeed,
	double *pX,
	double *pY,
	double *pZ,
	double *pVX,
	double *pVY,
	double *pVZ,
	double targetX,
	double targetY,
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

void social_force_model_randomNextStation(
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
