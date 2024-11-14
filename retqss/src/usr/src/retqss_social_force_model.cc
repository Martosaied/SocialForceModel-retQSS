#include "retqss_social_force_model.h"

#include "retqss/retqss_model_api.h"
#include "retqss/retqss_interface.hh"
#include "retqss/retqss_utilities.hh"

extern "C"
{

double norm(double aX, double aY, double aZ)
{
	return sqrt(aX*aX + aY*aY + aZ*aZ);
}

void repulsive_pedestrian_effect(
	double aX, double aY, double aZ,
	double bX, double bY, double bZ,
	double bVX, double bVY, double bVZ,
	double bSpeed,
	double *x, double *y, double *z
)
{
	double rX = aX - bX;
	double rY = aY - bY;
	double rZ = aZ - bZ;

	if (isnan(bVX) && isnan(bVY) && isnan(bVZ)) {
		bVX = 0;
		bVY = 0;
		bVZ = 0;
	}

	double rNorm = norm(rX, rY, rZ);

	int deltaT = 2;
	double bVNorm = norm(rX - deltaT * bVX, rY - deltaT * bVY, rZ - deltaT * bVZ);

	double b = sqrt(
		pow(rNorm + bVNorm, 2) 
		// - pow(bSpeed * deltaT, 2) TODO: revisar pq da negativo
	);

	double initialRepulsivePotential = 2.1; // TODO: make it a parameter
	double repulsivePotential = pow(initialRepulsivePotential, (-b/0.26));

	double w = 1; // TODO: esto es una funcion que da 1 o un numero de 0 a 1 dependiendo de la distancia

	printf("pow(rNorm + bVNorm, 2): %f\n", pow(rNorm + bVNorm, 2));
	printf("pow(bSpeed * deltaT, 2): %f\n", pow(bSpeed * deltaT, 2));
	printf("pow(rNorm + bVNorm, 2) - pow(bSpeed * deltaT, 2): %f\n", pow(rNorm + bVNorm, 2) - pow(bSpeed * deltaT, 2));
	printf("b: %f\n", b);

	*x = repulsivePotential * rX;
	*y = repulsivePotential * rY;
	*z = repulsivePotential * rZ;
}

void social_force_model_totalRepulsivePedestrianEffect(int particleID, double *pX, double *pY, double *pZ, double *pVX, double *pVY, double *pVZ, double *x, double *y, double *z)
{	
	double totalRepulsiveX = 0;
	double totalRepulsiveY = 0;
	double totalRepulsiveZ = 0;

	int index = (particleID-1)*3;
	for (int i = 0; i < 39; i++) {
		double repulsiveX, repulsiveY, repulsiveZ;
		repulsive_pedestrian_effect(pX[index], pY[index], pZ[index], pX[i*3], pY[i*3], pZ[i*3], pVX[i*3], pVY[i*3], pVZ[i*3], 1.2, &repulsiveX, &repulsiveY, &repulsiveZ);
		totalRepulsiveX += repulsiveX;
		totalRepulsiveY += repulsiveY;
		totalRepulsiveZ += repulsiveZ;
	}
	*x = totalRepulsiveX;
	*y = totalRepulsiveY;
	*z = totalRepulsiveZ;
}

void social_force_model_desiredDirection(
	double currentX,
	double currentY,
	double currentZ,
	double targetX,
	double targetY,
	double targetZ,
	double *desiredX,
	double *desiredY,
	double *desiredZ)
{
	// TODO: Agregar z a la norma
	double norm = sqrt((targetX - currentX) * (targetX - currentX) + (targetY - currentY) * (targetY - currentY));
	*desiredX = ((targetX - currentX) / norm);
	*desiredY = ((targetY - currentY) / norm);
	*desiredZ = currentZ;
}

void social_force_model_acceleration(
	int particleID,
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
	double *z
)
{	
	int index = (particleID-1)*3;
	double currentX = px[index];
	double currentY = py[index];
	double currentZ = pz[index];

	// The desired speed is gaussian distributed with mean 1.34 m/s and standard deviation 0.26 m/s
	double desiredSpeed = retQSS::random_normal(1.34, 0.26); // TODO: make it a parameter

	// The desired direction is given by the difference between the current position and the target position
	double desiredX, desiredY, desiredZ;
	social_force_model_desiredDirection(
		currentX, currentY, currentZ,
		targetX, targetY, targetZ,
		&desiredX, &desiredY, &desiredZ
	);

	// // The desired acceleration is the difference between the desired speed and the current speed
	desiredX = (desiredX*desiredSpeed);
	desiredY = (desiredY*desiredSpeed);
	desiredZ = (desiredZ*desiredSpeed);

	// Current velocity
	double currentVX = vx[index];
	double currentVY = vy[index];
	double currentVZ = vz[index];

	// The acceleration is the difference between the desired acceleration and the current acceleration
	// The acceleration has a relaxation time of 0.5 seconds
	// TODO: Missing difference between desired and actual
	double relaxationTime = 0.5; // TODO: make it a parameter
	*x = (desiredX - currentVX) / relaxationTime;
	*y = (desiredY - currentVY) / relaxationTime;
	*z = (desiredZ - currentVZ) / relaxationTime;
}


}
