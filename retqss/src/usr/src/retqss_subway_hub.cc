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

std::deque<int> objective_pathway = {10, 8, 107, 105, 103, 4, 2};
std::ofstream outputSubwayHubCSV("solution.csv");
bool startedSubwayHub = false;

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

Bool subway_hub_outputCSV_old_not_Working(double time,
	int N,
	double *x,
	int V,
	double *volumeConcentration,
    int recoveredCount,
    int infectionsCount)
{
	if(!startedSubwayHub) {
		outputSubwayHubCSV << "time";
		outputSubwayHubCSV << ",PX[1],PY[1],PS[1],PTS[1]";
		for(int i=1; i<=V; i++){
			outputSubwayHubCSV << ",VS[" << i << "],VC[" << i << "]";
		}
		outputSubwayHubCSV << ",recoveredCount,infectionsCount";
		outputSubwayHubCSV << std::endl;
		startedSubwayHub = true;
	} 
	outputSubwayHubCSV << std::fixed << std::setprecision(4) << time;
	int pStatus = (int) RETQSS()->particle_get_property(0, "status");
	int pTrackingStatus = (int) RETQSS()->particle_get_property(0, "trackingStatus");
	outputSubwayHubCSV << "," << x[0] << "," << x[N*3] << "," << pStatus << "," << pTrackingStatus;
	outputSubwayHubCSV << std::setprecision(12);
	for(int i=0; i < V; i++){
	    int vClosed = RETQSS()->volume_get_property(i+1, "isClosedSpace")?1:0;
		outputSubwayHubCSV << "," << vClosed << "," << volumeConcentration[i*3];
	}
	outputSubwayHubCSV << "," << recoveredCount << "," << infectionsCount;
	outputSubwayHubCSV << std::endl;
	outputSubwayHubCSV.flush();
	return true;
}

}