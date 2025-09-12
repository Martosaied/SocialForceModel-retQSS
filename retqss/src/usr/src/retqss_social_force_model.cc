#include "retqss_social_force_model.h"

#include "retqss/retqss_model_api.h"
#include "retqss/retqss_interface.hh"
#include "retqss/retqss_utilities.hh"
#include "retqss_utils.hh"

#include <cmath>
#include <fstream>
#include <set>
#include <chrono>
#include <ctime>
#include <cstddef>

std::list<Wall> walls;
std::map<int, std::vector<int>> neighboring_obstacles;

ModelParameters model_parameters;

std::ofstream sfm_outputCSV("solution.csv");
bool social_force_model_started = false;

#define IC_FILE "initial_conditions.ic"
#define PARAMS_FILE "parameters.config"
#define DEBUG 1

extern "C"
{

Bool social_force_model_setParameters() {
	model_parameters.PEDESTRIAN_A_1 = utils_getRealModelParameter("PEDESTRIAN_A_1", 2.1);
	model_parameters.PEDESTRIAN_B_1 = utils_getRealModelParameter("PEDESTRIAN_B_1", 0.3);
	model_parameters.PEDESTRIAN_A_2 = utils_getRealModelParameter("PEDESTRIAN_A_2", 2.1);
	model_parameters.PEDESTRIAN_B_2 = utils_getRealModelParameter("PEDESTRIAN_B_2", 0.3);
	model_parameters.PEDESTRIAN_R = utils_getRealModelParameter("PEDESTRIAN_R", 0.1);
	model_parameters.PEDESTRIAN_LAMBDA = utils_getRealModelParameter("PEDESTRIAN_LAMBDA", 0.3);
	model_parameters.PEDESTRIAN_IMPLEMENTATION = utils_getRealModelParameter("PEDESTRIAN_IMPLEMENTATION", 0);

	model_parameters.BORDER_IMPLEMENTATION = utils_getRealModelParameter("BORDER_IMPLEMENTATION", 0);
	model_parameters.BORDER_A = utils_getRealModelParameter("BORDER_A", 10);
	model_parameters.BORDER_B = utils_getRealModelParameter("BORDER_B", 0.7);
	model_parameters.BORDER_R = utils_getRealModelParameter("BORDER_R", 0.1);

	return true;
}

// Function to compute the dot product of two vectors
double dot_product(double aX, double aY, double bX, double bY) {
    return aX * bX + aY * bY;
}

// Function to compute the squared magnitude of a vector
double squared_magnitude(double aX, double aY) {
    return aX * aX + aY * aY;
}

void closest_point_on_segment(double pX, double pY, double aX, double aY, double bX, double bY, double *x, double *y) {
    double APx = pX - aX;
    double APy = pY - aY;
    double ABx = bX - aX;
    double ABy = bY - aY;
    
    double AB_squared = squared_magnitude(ABx, ABy);
    if (AB_squared == 0.0) {
        *x = aX;
        *y = aY;
        return; // A and B are the same point
    }
    
    double projection_scalar = dot_product(APx, APy, ABx, ABy) / AB_squared;
    
    if (projection_scalar < 0.0) {
        *x = aX;
        *y = aY;
    } else if (projection_scalar > 1.0) {
        *x = bX;
        *y = bY;
    } else {
        *x = aX + projection_scalar * ABx;
        *y = aY + projection_scalar * ABy;
    }
}

double vector_norm(double aX, double aY, double aZ)
{
	return sqrt(aX*aX + aY*aY + aZ*aZ);
}

void social_force_model_repulsiveBorderEffect(
	double A,
	double B,
	double ra,
	int particleID,
	double *x,
	double *y,
	double *z
) {
	double pX, pY, pZ;
	retQSS_particle_currentPosition(particleID, &pX, &pY, &pZ);

	for (Wall wall : walls) {
		double from_x = wall.from_x;
		double from_y = wall.from_y;
		double to_x = wall.to_x;
		double to_y = wall.to_y;

		double closest_x, closest_y;
		closest_point_on_segment(pX, pY, from_x, from_y, to_x, to_y, &closest_x, &closest_y);

		double deltay = closest_y - pY;
		double deltax = closest_x - pX;

		double distanceab = sqrt(deltax*deltax + deltay*deltay);

		double normalizedY = (pY - closest_y) / distanceab;
		double fy = A*exp((ra-distanceab)/B)*normalizedY;
	
		double normalizedX = (pX - closest_x) / distanceab;
		double fx = A*exp((ra-distanceab)/B)*normalizedX;

		*x += fx;
		*y += fy;
		*z = 0;
	}
	
}

void social_force_model_squareNearestPoint(
	int volumeID,
	double pointX,
	double pointY,
	double radius,
	double *x,
	double *y,
	double *z
) {
	double centroidX, centroidY, centroidZ;
	retQSS_volume_centroid(volumeID, &centroidX, &centroidY, &centroidZ);

    double qx = (pointX - centroidX) / radius;
    double qy = (pointY - centroidY) / radius;

    double f = std::max(std::abs(qx), std::abs(qy));
    double intersectX = qx/f;
    double intersectY = qy/f;
    *x = intersectX * radius + centroidX;
    *y = intersectY * radius + centroidY;
    *z = 0;
}

void social_force_model_nearestPoint(
	int volumeID,
	double pointX,
	double pointY,
	double radius,
	double *x,
	double *y,
	double *z)
{
	double borderX, borderY, borderZ;
	retQSS_volume_centroid(volumeID, &borderX, &borderY, &borderZ);

	double m = (borderY - pointY) / (borderX - pointX);
	double b = borderY - m * borderX;
	double h = borderX;
	double k = borderY;
	double r = radius;

	double A = 1 + m*m;
	double B = 2 * m * (b - k) - 2 * h;
	double C = h*h + (b - k)*(b - k) - r*r;

	// Discriminante
	double D = B*B - 4 * A * C;

	double sqrt_D = sqrt(D);
	double x1 = (-B + sqrt_D) / (2 * A);
	double x2 = (-B - sqrt_D) / (2 * A);
	double y1 = m * x1 + b;
	double y2 = m * x2 + b;

	// Calculate the distance to the point
	double distance1 = sqrt((x1 - pointX)*(x1 - pointX) + (y1 - pointY)*(y1 - pointY));
	double distance2 = sqrt((x2 - pointX)*(x2 - pointX) + (y2 - pointY)*(y2 - pointY));

	// Choose the nearest point
	if (distance1 < distance2) {
		*x = x1;
		*y = y1;
	} else {
		*x = x2;
		*y = y2;
	}

	*z = 0;
}

void social_force_model_onlyObstacleIterativeRepulsiveBorderEffect(
	int particleID,
	double cellEdgeLength,
	double *x,
	double *y,
	double *z
) {
	*x = 0;
	*y = 0;
	*z = 0;

	double pX, pY, pZ;
	retQSS_particle_currentPosition(particleID, &pX, &pY, &pZ);

	std::vector<int> volumes = utils_getArrayParameter("OBSTACLES");

	double A = model_parameters.BORDER_A;
	double B = model_parameters.BORDER_B;
	double r = model_parameters.BORDER_R;

	for (int volume : volumes) {
		double borderX, borderY, borderZ;
		social_force_model_squareNearestPoint(volume, pX, pY, cellEdgeLength/2, &borderX, &borderY, &borderZ);
		double distanceab = retQSS_volume_distanceToPoint(volume, pX, pY, pZ);

		double normalizedY = (pY - borderY) / distanceab;
		double fy = A*exp((r-distanceab)/B)*normalizedY;

		double normalizedX = (pX - borderX) / distanceab;
		double fx = A*exp((r-distanceab)/B)*normalizedX;

		*x += fx;
		*y += fy;
		*z = 0;
	}
}

void social_force_model_neighborsRepulsiveBorderEffect(
	double A,
	double B,
	double r,
	int particleID,
	double cellEdgeLength,
	double *x,
	double *y,
	double *z
) {
	*x = 0;
	*y = 0;
	*z = 0;

	double pX, pY, pZ;
	retQSS_particle_currentPosition(particleID, &pX, &pY, &pZ);

	std::vector<int> volumes = neighboring_obstacles[particleID];
	for (int volume : volumes) {
		double borderX, borderY, borderZ;
		social_force_model_squareNearestPoint(volume, pX, pY, cellEdgeLength/2, &borderX, &borderY, &borderZ);
		double distanceab = retQSS_volume_distanceToPoint(volume, pX, pY, pZ);

		double normalizedY = (pY - borderY) / distanceab;
		double fy = A*exp((r-distanceab)/B)*normalizedY;

		double normalizedX = (pX - borderX) / distanceab;
		double fx = A*exp((r-distanceab)/B)*normalizedX;

		*x += fx;
		*y += fy;
		*z = 0;
	}
}

void social_force_model_updateNeighboringVolumes(int particleID, int gridDivisions) {
	int volumeID = retQSS_particle_currentVolumeID(particleID);
	int upperVolumeID = volumeID + gridDivisions;
	int lowerVolumeID = volumeID - gridDivisions;
	int rightVolumeID = volumeID + 1;
	int leftVolumeID = volumeID - 1;
	int upperRightVolumeID = upperVolumeID + 1;
	int upperLeftVolumeID = upperVolumeID - 1;
	int lowerRightVolumeID = lowerVolumeID + 1;
	int lowerLeftVolumeID = lowerVolumeID - 1;

	std::vector<int> volumes = {
		upperVolumeID, lowerVolumeID, 
		rightVolumeID, leftVolumeID, 
		upperRightVolumeID, upperLeftVolumeID, lowerRightVolumeID, lowerLeftVolumeID, 
	};

	// Filter the ones that are not in the grid or not obstacles
	neighboring_obstacles[particleID] = {};

	for (int id : volumes) {
		if (id >= 1 && id <= gridDivisions*gridDivisions && retQSS_volume_getProperty(id, "isObstacle")) {
			neighboring_obstacles[particleID].push_back(id);
		}
	}
}

bool repulsive_pedestrian_effect(
    retQSS::ParticleNeighbor *neighbor,
    const std::vector<double> &args,
    Vector_3 &result)
{
	retQSS::Particle *p = neighbor->source_particle();
	retQSS::Particle *q = neighbor->neighbor_particle();

	int pID = p->get_ID() + 1;
	int qID = q->get_ID() + 1;

	if (pID == qID) {
		return false; // Skip if the source and neighbor particle are the same
	}

	double A_1 = model_parameters.PEDESTRIAN_A_1;
	double B_1 = model_parameters.PEDESTRIAN_B_1;

	double A_2 = model_parameters.PEDESTRIAN_A_2;
	double B_2 = model_parameters.PEDESTRIAN_B_2;

	double ra = model_parameters.PEDESTRIAN_R;
	double rb = model_parameters.PEDESTRIAN_R;
	double rab = ra + rb;

	double aX, aY, aZ, bX, bY, bZ;

    retQSS_particle_currentPosition(pID, &aX, &aY, &aZ);
    retQSS_particle_currentPosition(qID, &bX, &bY, &bZ);

	double targetX = args[0];
	double targetY = args[1];

	double deltax = bX - aX;
	double deltay = bY - aY;
	double distanceab = sqrt(deltax*deltax + deltay*deltay);

	double normalizedX = (aX - bX) / distanceab;
	double normalizedY = (aY - bY) / distanceab;

	double fx_1 = A_1*exp((rab-distanceab)/B_1)*normalizedX;
	double fy_1 = A_1*exp((rab-distanceab)/B_1)*normalizedY;

	double fx_2 = A_2*exp((rab-distanceab)/B_2)*normalizedX;
	double fy_2 = A_2*exp((rab-distanceab)/B_2)*normalizedY;

	double lambda = model_parameters.PEDESTRIAN_LAMBDA;
	double desiredX, desiredY, desiredZ;
	social_force_model_desiredDirection(
		aX, aY, aZ,
		targetX, targetY, 0,
		&desiredX, &desiredY, &desiredZ
	);
	double cos_phi = -(normalizedX*desiredX) - (normalizedY*desiredY);
	double area = lambda + (1-lambda)*((1+cos_phi)/2);
	
	result = Vector_3((fx_1 * area) + fx_2, (fy_1 * area) + fy_2, 0);

	return true;
}

void social_force_model_volumeBasedRepulsivePedestrianEffect(
	int particleID,
	double targetX,
	double targetY,
	double *totalRepulsiveX,
	double *totalRepulsiveY,
	double *totalRepulsiveZ
) {

	double A_1 = model_parameters.PEDESTRIAN_A_1;
	double B_1 = model_parameters.PEDESTRIAN_B_1;

	double A_2 = model_parameters.PEDESTRIAN_A_2;
	double B_2 = model_parameters.PEDESTRIAN_B_2;

	double ra = model_parameters.PEDESTRIAN_R;
	double rb = model_parameters.PEDESTRIAN_R;
	double rab = ra + rb;

	*totalRepulsiveX = 0;
	*totalRepulsiveY = 0;
	*totalRepulsiveZ = 0;

	int currentVolumeID = retQSS_particle_currentVolumeID(particleID);
	int particlesInVolume = retQSS_volume_countParticlesInside(currentVolumeID);

	for (int i = 1; i <= particlesInVolume; i++) {
		int neighborID = retQSS_volume_IDOfParticleInside(currentVolumeID, i);
		double aX, aY, aZ, bX, bY, bZ;
		retQSS_particle_currentPosition(particleID, &aX, &aY, &aZ);
		retQSS_particle_currentPosition(neighborID, &bX, &bY, &bZ);

		double deltax = bX - aX;
		double deltay = bY - aY;
		double distanceab = sqrt(deltax*deltax + deltay*deltay);

		double normalizedX = (aX - bX) / distanceab;
		double normalizedY = (aY - bY) / distanceab;

		double fx_1 = A_1*exp((rab-distanceab)/B_1)*normalizedX;
		double fy_1 = A_1*exp((rab-distanceab)/B_1)*normalizedY;

		double fx_2 = A_2*exp((rab-distanceab)/B_2)*normalizedX;
		double fy_2 = A_2*exp((rab-distanceab)/B_2)*normalizedY;

		double lambda = model_parameters.PEDESTRIAN_LAMBDA;
		double desiredX, desiredY, desiredZ;
		social_force_model_desiredDirection(
			aX, aY, aZ,
			targetX, targetY, 0,
			&desiredX, &desiredY, &desiredZ
		);
		double cos_phi = -(normalizedX*desiredX) - (normalizedY*desiredY);
		double area = lambda + (1-lambda)*((1+cos_phi)/2);

		*totalRepulsiveX += (fx_1 * area) + fx_2;
		*totalRepulsiveY += (fy_1 * area) + fy_2;
		*totalRepulsiveZ = 0;
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

Bool social_force_model_outputCSV(double time,
	int N,
	double *x)
{
	if(!social_force_model_started) {
		sfm_outputCSV << "time";
		for(int i=1; i<=N; i++) {
			sfm_outputCSV << ",PX[" << i << "],PY[" << i << "],VX[" << i << "],VY[" << i << "],PS[" << i << "]";
		}
		sfm_outputCSV << std::endl;
		social_force_model_started = true;
	} 
	sfm_outputCSV << std::fixed << std::setprecision(4) << time;
	for(int i=0; i < N; i++){
	    int pType = (int) RETQSS()->particle_get_property(i, "type");
		double y = x[(i+N)*3];
		double vx = x[(i+2*N)];
		double vy = x[(i+3*N)];
		sfm_outputCSV << "," << x[i*3] << "," << y << "," << vx << "," << vy << "," << pType;
	}
	sfm_outputCSV << std::endl;
	sfm_outputCSV.flush();
	return true;
}

Bool social_force_model_notQSS_outputCSV(double time,
	int N,
	int *groupIDs,
	double *x)
{
	if(!social_force_model_started) {
		sfm_outputCSV << "time";
		for(int i=1; i<=N; i++) {
			sfm_outputCSV << ",PX[" << i << "],PY[" << i << "],VX[" << i << "],VY[" << i << "],PS[" << i << "]";
		}
		sfm_outputCSV << std::endl;
		social_force_model_started = true;
	} 
	sfm_outputCSV << std::fixed << std::setprecision(4) << time;
	for(int i=0; i < N; i++){
	    int pType = groupIDs[i];
		double y = x[(i+N)*3];
		double vx = x[(i+2*N)];
		double vy = x[(i+3*N)];
		sfm_outputCSV << "," << x[i*3] << "," << y << "," << vx << "," << vy << "," << pType;
	}
	sfm_outputCSV << std::endl;
	sfm_outputCSV.flush();
	return true;
}

Bool social_force_model_setUpParticles(
    int N,
    double cellEdgeLength,
    int gridDivisions,
    double *x)
{
	std::ofstream output(IC_FILE);
	for(int p = 0; p < N; p++)
	{
		for(int c=0;c<6;c++){
			output << x[(p+c*N)*3] << " ";
		}
		double px = x[p*3] / cellEdgeLength;
		double py = x[(p+N)*3] / cellEdgeLength;
		int volumeID = ((int) px) * gridDivisions + ((int) py) + 1;
	    output << " " << volumeID << std::endl;
	}
	output.close();
	retQSS_particle_setUpFromFile(N, IC_FILE, "social_force_model");
    return true;
}

Bool social_force_model_setUpWalls() {
	std::string walls_str = utils_getParameter("WALLS");
	
	std::string item;
	std::stringstream ss(walls_str);
	while (std::getline(ss, item, ',')) {
		std::stringstream iss(item);
		std::string s;
		Wall wall;
		std::getline(iss, s, '/');
		wall.from_x = std::stod(s);
		std::getline(iss, s, '/');
		wall.from_y = std::stod(s);
		std::getline(iss, s, '/');
		wall.to_x = std::stod(s);
		std::getline(iss, s, '/');
		wall.to_y = std::stod(s);

		walls.push_back(wall);
	}

	return true;
}



}