#include "retqss_subway_hub.h"

#include "retqss/retqss_model_api.h"
#include "retqss/retqss_interface.hh"
#include "retqss/retqss_utilities.hh"
#include "retqss_utils.hh"
#include "retqss_pathways.h"

#include <cmath>
#include <fstream>
#include <set>
#include <chrono>
#include <ctime>
#include <cstddef>
#include <algorithm>

std::deque<int> objective_pathway = {10, 8, 98, 96, 94, 4, 2};

extern "C"
{

Bool subway_hub_setUpParticleMovement(int N) {
    for (int i = 1; i <= N; i++) {
        if (retQSS_particle_getProperty(i, "isObjective") == 1) {
            pathways_setPathway(i, objective_pathway);
        }
    }
	return true;
}

}