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

void social_force_model_pedestrianAcceleration(
	double pX,
	double pY,
	double pZ,
	double vX,
	double vY,
	double vZ,
	double targetX,
	double targetY,
	double targetZ,
	double *x,
	double *y,
	double *z
)
{	
	double currentX = pX;
	double currentY = pY;
	double currentZ = pZ;

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
	double currentVX = vX;
	double currentVY = vY;
	double currentVZ = vZ;

	// The acceleration is the difference between the desired acceleration and the current acceleration
	// The acceleration has a relaxation time of 0.5 seconds
	// TODO: Missing difference between desired and actual
	double relaxationTime = 0.5; // TODO: make it a parameter
	*x = (desiredX - currentVX) / relaxationTime;
	*y = (desiredY - currentVY) / relaxationTime;
	*z = (desiredZ - currentVZ) / relaxationTime;
}

void social_force_model_repulsivePedestrianEffect(
	double pX1, double pY1, double pZ1,
	double pX2, double pY2, double pZ2,
	double vX2, double vY2, double vZ2,
	double *x, double *y, double *z
)
{
	double rX = pX1 - pX2;
	double rY = pY1 - pY2;
	double rZ = pZ1 - pZ2;

	if (isnan(vX2) && isnan(vY2) && isnan(vZ2)) {
		vX2 = 0;
		vY2 = 0;
		vZ2 = 0;
	}

	double rNorm = norm(rX, rY, rZ);

	int deltaT = 2;
	double bVNorm = norm(rX - deltaT * vX2, rY - deltaT * vY2, rZ - deltaT * vZ2);

	double b = sqrt(
		pow(rNorm + bVNorm, 2) 
		// - pow(bSpeed * deltaT, 2) TODO: revisar pq da negativo
	);

	double initialRepulsivePotential = 2.1; // TODO: make it a parameter
	double repulsivePotential = pow(initialRepulsivePotential, (-b/0.26));

	double w = 1; // TODO: esto es una funcion que da 1 o un numero de 0 a 1 dependiendo de la distancia


	*x = repulsivePotential * rX;
	*y = repulsivePotential * rY;
	*z = repulsivePotential * rZ;
}

}
