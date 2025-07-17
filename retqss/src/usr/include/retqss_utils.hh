#ifndef _RETQSS_UTILS_H_
#define _RETQSS_UTILS_H_

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


int utils_setDebugLevel(int level);

Bool utils_isDebugLevelEnabled(int level);

int utils_debug(int level, double time, const char *format, int int1, int int2, double double1, double double2);

double utils_arrayGet(double *array, int index);

Bool utils_arraySet(double *array, int index, double value);

int utils_getIntegerModelParameter(const char *name, int defaultValue);

double utils_getRealModelParameter(const char *name, double defaultValue);

Bool utils_isInArrayParameter(const char *name, int value);

#ifdef __cplusplus

std::string utils_getParameter(const char *name);

}
#endif

#endif
