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
	double *x, double *y, double *z
)
{
	// double deltax = 0.001;
	// double deltaT2 = 0.002;
	// double rabmod = sqrt(aX*aX + aY*aY);
	// double rabmodx = sqrt((aX+deltax)*(aX+deltax) + aY*aY);
	// double rabmody = sqrt(aX*aX + (aY+deltax)*(aY+deltax));

	// double rabx = bX - aX;
	// double raby = bY - aY;

	// double theta = atan2(raby, rabx);
	// double thetax = atan2(raby, rabx+deltax);
	// double thetay = atan2(raby+deltax, rabx);

	// double vb = sqrt(bVX*bVX + bVY*bVY);

	// double sigma = 0.3;

	// double root = sqrt(rabmod*rabmod-2.0*vb*deltaT2*rabmod*cos(theta)+vb*vb*deltaT2*deltaT2);
	// double rootx = sqrt(rabmodx*rabmodx-2.0*vb*deltaT2*rabmodx*cos(thetax)+vb*vb*deltaT2*deltaT2);
	// double rooty = sqrt(rabmody*rabmody-2.0*vb*deltaT2*rabmody*cos(thetay)+vb*vb*deltaT2*deltaT2);
	// double b = sqrt(rabmod*rabmod+2.0*rabmod*root+root*root-vb*vb*deltaT2*deltaT2)/2.0;
	// double bx = sqrt(rabmodx*rabmodx+2.0*rabmodx*rootx+rootx*rootx-vb*vb*deltaT2*deltaT2)/2.0;
	// double by = sqrt(rabmody*rabmody+2.0*rabmody*rooty+rooty*rooty-vb*vb*deltaT2*deltaT2)/2.0;
	// double e = exp(-b/sigma);
	// double expx = exp(-bx/sigma);
	// double expy = exp(-by/sigma);
	// double fx = -2.1*(expx-e)/deltax;
	// double fy = -2.1*(expy-e)/deltax;

	// double phi = 100.0*2.0*3.1415926/360.0;
	// double c = 0.5;
	// if (-aX*fx >= sqrt(fx*fx+fy*fy)*cos(phi)) {
	// 	fx = fx;
	// 	fy = fy;
	// } else {
	// 	fx = fx*c;
	// 	fy = fy*c;
	// }

	double A = 4;
	double B = 0.2;

	double ra = 0.01;
	double rb = 0.01;
	double rab = ra + rb;

	double deltax = bX - aX;
	double deltay = bY - aY;
	double distanceab = sqrt(deltax*deltax + deltay*deltay);

	double normalizedX = deltax / distanceab;
	double normalizedY = deltay / distanceab;

	double fx = A*exp((rab-distanceab)/B)*normalizedX;
	double fy = A*exp((rab-distanceab)/B)*normalizedY;

	*x = fx;
	*y = fy;
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
	double *x, 
	double *y, 
	double *z
)
{	
	double totalRepulsiveX = 0;
	double totalRepulsiveY = 0;
	double totalRepulsiveZ = 0;

	int index = (particleID-1)*3;
	for (int i = 0; i < 99; i++) {
		if (i == particleID-1) continue;
		double repulsiveX, repulsiveY, repulsiveZ;
		repulsive_pedestrian_effect(pX[index], pY[index], pZ[index], pX[i*3], pY[i*3], pZ[i*3], pVX[i*3], pVY[i*3], pVZ[i*3], desiredSpeed[i], &repulsiveX, &repulsiveY, &repulsiveZ);
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
	int index = (particleID-1)*3;
	double aX = pX[index];
	double aY = pY[index];
	double b_inf = 0;
	double b_sup = 1;

	double A = 4;
	double B = 0.2;

	double ra = 0.01;

	double deltax = aX;
	double deltay = b_inf - aY;
	double distanceab = sqrt(deltax*deltax + deltay*deltay);

	double normalizedX = deltax / distanceab;
	double normalizedY = deltay / distanceab;

	double fx_inf = A*exp((ra-distanceab)/B)*normalizedX;
	double fy_inf = A*exp((ra-distanceab)/B)*normalizedY;

	deltax = aX;
	deltay = b_sup - aY;
	distanceab = sqrt(deltax*deltax + deltay*deltay);

	normalizedX = deltax / distanceab;
	normalizedY = deltay / distanceab;

	double fx_sup = A*exp((ra-distanceab)/B)*normalizedX;
	double fy_sup = A*exp((ra-distanceab)/B)*normalizedY;

	*x = fx_inf + fx_sup;
	*y = fy_inf + fy_sup;
	*z = 0;
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
