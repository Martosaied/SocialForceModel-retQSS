#include "retqss_social_force_model.h"

#include "retqss/retqss_model_api.h"
#include "retqss/retqss_interface.hh"
#include "retqss/retqss_utilities.hh"

extern "C"
{

double vector_norm(double aX, double aY, double aZ)
{
	return sqrt(aX*aX + aY*aY + aZ*aZ);
}

void repulsive_pedestrian_effect(
	double aX, double aY, double aZ,
	double bX, double bY, double bZ,
	double bVX, double bVY, double bVZ,
	double bSpeed,
	double targetX,
	double targetY,
	double *x, double *y, double *z
)
{
	double A = 2.1;
	double B = 0.3;

	double ra = 0.1;
	double rb = 0.1;
	double rab = ra + rb;

	double deltax = bX - aX;
	double deltay = bY - aY;
	double distanceab = sqrt(deltax*deltax + deltay*deltay);

	double normalizedX = (aX - bX) / distanceab;
	double normalizedY = (aY - bY) / distanceab;

	double fx = A*exp((rab-distanceab)/B)*normalizedX;
	double fy = A*exp((rab-distanceab)/B)*normalizedY;

	double lambda = 0.3;
	double desiredX, desiredY, desiredZ;
	social_force_model_desiredDirection(
		aX, aY, aZ,
		targetX, targetY, 0,
		&desiredX, &desiredY, &desiredZ
	);
	double cos_phi = -normalizedX*desiredX - normalizedY*desiredY;
	double area = lambda + (1-lambda)*((1+cos_phi)/2);

	*x = fx*area;
	*y = fy*area;
	*z = 0;
}

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
	double *z
)
{	
	double totalRepulsiveX = 0;
	double totalRepulsiveY = 0;
	double totalRepulsiveZ = 0;

	int index = (particleID-1)*3;
	for (int i = 0; i < 299; i++) {
		if (i == particleID-1) continue;
		double repulsiveX, repulsiveY, repulsiveZ;
		repulsive_pedestrian_effect(pX[index], pY[index], pZ[index], pX[i*3], pY[i*3], pZ[i*3], pVX[i*3], pVY[i*3], pVZ[i*3], desiredSpeed[i], targetX, targetY, &repulsiveX, &repulsiveY, &repulsiveZ);
		totalRepulsiveX += repulsiveX;
		totalRepulsiveY += repulsiveY;
		totalRepulsiveZ += repulsiveZ;
	}
	*x = totalRepulsiveX;
	*y = totalRepulsiveY;
	*z = totalRepulsiveZ;	
}

void social_force_model_totalRepulsiveBorderEffect(
	int particleID,
	double pX[1],
	double pY[1],
	double pZ[1],
	double *x,
	double *y,
	double *z
)
{
	*x = 0;
	*y = 0;
	*z = 0;
	for (int i = 1; i < 400; i++) {
		if (retQSS_volume_getProperty(i, "isObstacle")) {
			int index = (particleID-1)*3;
			double aY = pY[index];
			double aX = pX[index];

			double borderX, borderY, borderZ;

			// Calculate the forces from the centroid to be even from all sides
			retQSS_volume_centroid(i, &borderX, &borderY, &borderZ);

			double A = 10;
			double B = 0.2;

			double ra = 0.01;


			double deltay = borderY - aY;
			double deltax = borderX - aX;

			double distanceab = sqrt(deltax*deltax + deltay*deltay);

			double normalizedY = (aY - borderY) / distanceab;
			double fy = A*exp((ra-distanceab)/B)*normalizedY;
		
			double normalizedX = (aX - borderX) / distanceab;
			double fx = A*exp((ra-distanceab)/B)*normalizedX;

			*x += fx;
			*y += fy;
			*z = 0;
		}
	}
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
	double norm = vector_norm((targetX - currentX), (targetY - currentY), (targetZ - currentZ));
	*desiredX = ((targetX - currentX) / norm);
	*desiredY = ((targetY - currentY) / norm);
	*desiredZ = currentZ;
}

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
	double *z
)
{	
	int index = (particleID-1)*3;
	double currentX = px[index];
	double currentY = py[index];
	double currentZ = pz[index];

	// The desired speed is gaussian distributed with mean 1.34 m/s and standard deviation 0.26 m/s
	double desiredSpeedValue = desiredSpeed[particleID];

	// The desired direction is given by the difference between the current position and the target position
	double desiredX, desiredY, desiredZ;
	social_force_model_desiredDirection(
		currentX, currentY, currentZ,
		targetX, targetY, targetZ,
		&desiredX, &desiredY, &desiredZ
	);

	// // The desired acceleration is the difference between the desired speed and the current speed
	desiredX = (desiredX*desiredSpeedValue);
	desiredY = (desiredY*desiredSpeedValue);
	desiredZ = (desiredZ*desiredSpeedValue);

	// Current velocity
	double currentVX = vx[index];
	double currentVY = vy[index];
	double currentVZ = vz[index];

	// The acceleration is the difference between the desired acceleration and the current acceleration
	// The acceleration has a relaxation time of 0.5 seconds
	double relaxationTime = 1/0.5; // TODO: make it a parameter
	*x = (desiredX - currentVX) * relaxationTime;
	*y = (desiredY - currentVY) * relaxationTime;
	*z = 0;

}


}
