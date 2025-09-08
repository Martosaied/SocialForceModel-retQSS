#include "retqss_utils.hh"

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
std::map<std::string, std::string> cache_parameters;
int a = 0;

#define IC_FILE "initial_conditions.ic"
#define PARAMS_FILE "parameters.config"
#define DEBUG 1

extern "C"
{

int utils_setDebugLevel(int level)
{
	debugLevel = level;
	return level;
}

Bool utils_isDebugLevelEnabled(int level)
{
	return level <= debugLevel;
}

int utils_debug(int level, double time, const char *format, int int1, int int2, double double1, double double2)
{
	if (utils_isDebugLevelEnabled(level)) {
	    std::time_t current_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	    char * ct = std::ctime(&current_time);
	    ct[strcspn(ct, "\n")] = '\0';
	    printf("[%s] (t=%.2f) ", ct, time);
	    printf(format, (int) int1, (int) int2, double1, double2);
	    printf("\n");
	}
	return level;
}

double utils_arrayGet(double *array, int index)
{
	return array[index-1];
}

Bool utils_arraySet(double *array, int index, double value)
{
	array[index-1] = value;
	return true;
}


std::string utils_getParameter(const char *name) {
	if (cache_parameters.find(name) != cache_parameters.end()) {
		return cache_parameters[name];
	}

	std::cout << "Parameter " << name << " not found" << std::endl;

	// Read the parameters from the file
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
			cache_parameters[name] = value;
			return value;
		}
	  }
	}


	cache_parameters[name] = std::string("");
	return std::string("");
}

Bool utils_isInArrayParameter(const char *name, int value) {
	std::string parameter = utils_getParameter(name);
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

std::vector<int> utils_getArrayParameter(const char *name) {
	std::string parameter = utils_getParameter(name);
	std::vector<int> array;
	if (parameter.empty()) {
		return array;
	}
	std::stringstream ss(parameter);
	std::string item;
	while (std::getline(ss, item, ',')) {
		array.push_back(std::stoi(item));
	}
	return array;
}

int utils_getIntegerModelParameter(const char *name, int defaultValue) {
	std::string value = utils_getParameter(name);
	return value == "" ? defaultValue : std::stoi(value);
}

double utils_getRealModelParameter(const char *name, double defaultValue) {
	std::string value = utils_getParameter(name);
	return value == "" ? defaultValue : std::stof(value);
}

}