#include "retqss_social_force_model.hh"

#include "retqss/retqss_model_api.h"
#include "retqss/retqss_interface.hh"
#include "retqss/retqss_utilities.hh"

#include <cmath>
#include <fstream>
#include <set>
#include <chrono>
#include <ctime>
#include <cstddef>

int debugLevel;
std::unordered_map<std::string, std::string> parameters;

std::ofstream outputCSV("solution.csv");
bool started = false;

#define IC_FILE "initial_conditions.ic"
#define PARAMS_FILE "parameters.config"
#define DEBUG 1

extern "C"
{

int social_force_model_setDebugLevel(int level)
{
	debugLevel = level;
	return level;
}

Bool social_force_model_isDebugLevelEnabled(int level)
{
	return level <= debugLevel;
}

int social_force_model_debug(int level, double time, const char *format, int int1, int int2, double double1, double double2)
{
	if (social_force_model_isDebugLevelEnabled(level)) {
	    std::time_t current_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	    char * ct = std::ctime(&current_time);
	    ct[strcspn(ct, "\n")] = '\0';
	    printf("[%s] (t=%.2f) ", ct, time);
	    printf(format, (int) int1, (int) int2, double1, double2);
	    printf("\n");
	}
	return level;
}

double social_force_model_arrayGet(double *array, int index)
{
	return array[index-1];
}

Bool social_force_model_arraySet(double *array, int index, double value)
{
	array[index-1] = value;
	return true;
}


std::string social_force_model_getParameter(const char *name) {
	if (parameters.find(name) != parameters.end()) {
		return parameters[name];
	}

	std::ifstream is_file(PARAMS_FILE);
	std::string line;
	while( std::getline(is_file, line) )
	{
	  std::istringstream is_line(line);
	  std::string key;
	  if( std::getline(is_line, key, '=') && key == name)
	  {
	    std::string value;
	    if( std::getline(is_line, value) ){
			parameters[name] = value;
			return value;
		}
	  }
	}

	parameters[name] = std::string("");
	return std::string("");
}

Bool social_force_model_isInArrayParameter(const char *name, int value) {
	std::string parameter = social_force_model_getParameter(name);
	// Convert the string to an array of integers
	std::vector<int> array;
	std::stringstream ss(parameter);
	std::string item;
	while (std::getline(ss, item, ',')) {
		array.push_back(std::stoi(item));
	}

	if (array.empty()) {
		return false;
	} else {
		return std::find(array.begin(), array.end(), value) != array.end();
	}
}

int social_force_model_getIntegerModelParameter(const char *name, int defaultValue) {
	std::string value = social_force_model_getParameter(name);
	return value == "" ? defaultValue : std::stoi(value);
}

double social_force_model_getRealModelParameter(const char *name, double defaultValue) {
	std::string value = social_force_model_getParameter(name);
	return value == "" ? defaultValue : std::stof(value);
}


double vector_norm(double aX, double aY, double aZ)
{
	return sqrt(aX*aX + aY*aY + aZ*aZ);
}

bool repulsive_pedestrian_effect(
    retQSS::ParticleNeighbor *neighbor,
    const std::vector<double> &args,
    Vector_3 &result)
{
	retQSS::Particle *p = neighbor->source_particle();
	retQSS::Particle *q = neighbor->neighbor_particle();

	if (p->get_ID() == 0 || q->get_ID() == 0) {
		return false; // Skip the source particle
	}

	double A = 2.1;
	double B = 0.3;

	double ra = 0.1;
	double rb = 0.1;
	double rab = ra + rb;

	double aX, aY, aZ, bX, bY, bZ, bVX, bVY, bVZ;

    retQSS_particle_currentPosition(p->get_ID(), &aX, &aY, &aZ);
    retQSS_particle_currentPosition(q->get_ID(), &bX, &bY, &bZ);
    retQSS_particle_currentVelocity(q->get_ID(), &bVX, &bVY, &bVZ);

	double targetX = args[0];
	double targetY = args[1];


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
	
	Vector_3 force = Vector_3(fx*area, fy*area, 0);
	result += force;
	return true;
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
		// repulsive_pedestrian_effect(pX[index], pY[index], pZ[index], pX[i*3], pY[i*3], pZ[i*3], pVX[i*3], pVY[i*3], pVZ[i*3], desiredSpeed[i], targetX, targetY, &repulsiveX, &repulsiveY, &repulsiveZ);
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

			double A = 100;
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

Bool social_force_model_outputCSV(double time,
	int N,
	double *x)
{
	if(!started) {
		outputCSV << "time";
		for(int i=1; i<=N; i++) {
			outputCSV << ",PX[" << i << "],PY[" << i << "],PS[" << i << "]";
		}
		outputCSV << std::endl;
		started = true;
	} 
	outputCSV << std::fixed << std::setprecision(4) << time;
	for(int i=0; i < N; i++){
	    int pType = (int) RETQSS()->particle_get_property(i, "type");
		outputCSV << "," << x[i*3] << "," << x[(i+N)*3] << "," << pType;
	}
	outputCSV << std::endl;
	outputCSV.flush();
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
	retQSS_particle_setUpFromFile(N, IC_FILE, "indirect_infection");
    return true;
}

}
