model subway_hub

import retQSS;
import retQSS_utils;

import retQSS_social_force_model;
import retQSS_social_force_model_utils;
import retQSS_social_force_model_params;
import retQSS_social_force_model_types;

import retQSS_subway_hub;
import retQSS_pathways;

import retQSS_covid19;
import retQSS_covid19_utils;
import retQSS_covid19_fsm;

/*
  This section loads parameters and constants from the parameters.config file
*/

constant Integer
	N = 1,
	GRID_DIVISIONS = 10;

// Initial conditions parameters
parameter Integer
	RANDOM_SEED = getIntegerModelParameter("RANDOM_SEED", 0),
	FORCE_TERMINATION_AT = getRealModelParameter("FORCE_TERMINATION_AT", 40), // 40 minutes
	PEDESTRIAN_IMPLEMENTATION = getIntegerModelParameter("PEDESTRIAN_IMPLEMENTATION", 0),
	BORDER_IMPLEMENTATION = getIntegerModelParameter("BORDER_IMPLEMENTATION", 0),
	CONVEYOR_BELT_EFFECT = getIntegerModelParameter("CONVEYOR_BELT_EFFECT", 0),
	SOCIAL_FORCE_MODEL = getIntegerModelParameter("SOCIAL_FORCE_MODEL", 0),

	INITIAL_EXPOSED = getIntegerModelParameter("INITIAL_EXPOSED", 0),
	INITIAL_INFECTED = getIntegerModelParameter("INITIAL_INFECTED", 9);

// Output delta time parameter
parameter Real
	DEFAULT_SPEED = getRealModelParameter("DEFAULT_SPEED", 1.34), // 1.34 m/s or 80.4 m/min
	OUTPUT_UPDATE_DT = getRealModelParameter("OUTPUT_UPDATE_DT", 1), // 0.5 seconds
	PROGRESS_UPDATE_DT = getRealModelParameter("PROGRESS_UPDATE_DT", 0.5), // 0.5 seconds
	MOTIVATION_UPDATE_DT = getRealModelParameter("MOTIVATION_UPDATE_DT", 0.1), // 0.5 seconds
	DESTINATION_UPDATE_DT = getRealModelParameter("DESTINATION_UPDATE_DT", 600), // 10 minutes
	OBJECTIVE_SUBWAY_HUB_DT = getRealModelParameter("OBJECTIVE_SUBWAY_HUB_DT", 600), // 10 minutes in each subway
	SPEED_MU = getRealModelParameter("SPEED_MU", 1.34), // 1.34 m/s or 80.4 m/min
	SPEED_SIGMA = getRealModelParameter("SPEED_SIGMA", 0.26), // 0.26 m/s or 15.6 m/min
	FROM_Y = getRealModelParameter("FROM_Y", 0.0),
	TO_Y = getRealModelParameter("TO_Y", 20.0);

// Disease population and evolution parameters
parameter Real
	YOUNG_PROB = getRealModelParameter("YOUNG_PROB", 0.25),
	ADULT_PROB = getRealModelParameter("ADULT_PROB", 0.5),

	YOUNG_SYMPTOMATIC_PROB = getRealModelParameter("YOUNG_SYMPTOMATIC_PROB", 0.0),
	ADULT_SYMPTOMATIC_PROB = getRealModelParameter("ADULT_SYMPTOMATIC_PROB", 0.0),
	OLD_SYMPTOMATIC_PROB = getRealModelParameter("OLD_SYMPTOMATIC_PROB", 0.0),

	YOUNG_DEATH_PROB = getRealModelParameter("YOUNG_DEATH_PROB", 0.0),
	ADULT_DEATH_PROB = getRealModelParameter("ADULT_DEATH_PROB", 0.0),
	OLD_DEATH_PROB = getRealModelParameter("OLD_DEATH_PROB", 0.0),

	INCUBATION_TIME_M = 	getRealModelParameter("INCUBATION_TIME_M", 140832),		// https://bmjopen.bmj.com/content/10/8/e039652
	INCUBATION_TIME_S = 	getRealModelParameter("INCUBATION_TIME_S", 43200),		//
	PRESYMPTOMATIC_TIME_M = getRealModelParameter("PRESYMPTOMATIC_TIME_M", 172800),		// https://www.researchgate.net/publication/344358423_SARS-CoV-2_transmission_dynamics_should_inform_policy/link/5fb1970492851cf24cd57da1/download
	PRESYMPTOMATIC_TIME_S = getRealModelParameter("PRESYMPTOMATIC_TIME_S", 43200),	// CI95% [1, 3]
	SYMPTOMATIC_TIME_M = 	getRealModelParameter("SYMPTOMATIC_TIME_M", 864000),		// https://www.researchgate.net/publication/344358423_SARS-CoV-2_transmission_dynamics_should_inform_policy/link/5fb1970492851cf24cd57da1/download
	SYMPTOMATIC_TIME_S = 	getRealModelParameter("SYMPTOMATIC_TIME_S", 129600);		// CI95% [7, 13]

// Disease propagation parameters
parameter Real
	CLOSE_CONTACT_PROB = getRealModelParameter("CLOSE_CONTACT_PROB", 0.0), //No close contact / p2p contagion
	PARTICLE_TO_PARTICLE_INFECTION_PROB = getRealModelParameter("PARTICLE_TO_PARTICLE_INFECTION_PROB", 0.4),
	SYMPTOMATIC_CONTAGION_PROB = getRealModelParameter("SYMPTOMATIC_CONTAGION_PROB", 1.0),
	PRESYMPTOMATIC_CONTAGION_PROB = getRealModelParameter("PRESYMPTOMATIC_CONTAGION_PROB", SYMPTOMATIC_CONTAGION_PROB), //same as symptomatic
	ASYMPTOMATIC_CONTAGION_PROB = getRealModelParameter("ASYMPTOMATIC_CONTAGION_PROB", SYMPTOMATIC_CONTAGION_PROB), //same as symptomatic

	VOLUME_CONCENTRATION_LAMBDA = getRealModelParameter("VOLUME_CONCENTRATION_LAMBDA", 3.36 / 60 / 60), // JLJ Table SI-4 (classroom / prepandemic scenario) = lambda (hourly to secondly conversion) 

	BREATHING_INTERVAL_TIME = getRealModelParameter("BREATHING_INTERVAL_TIME", 13.8 / 60), //~ 13.8 breaths per minute  https://my.clevelandclinic.org/health/articles/10881-vital-signs#:~:text=Respiratory%20rate%3A%20A%20person's%20respiratory,while%20resting%20is%20considered%20abnormal.

	SYMPTOMATIC_EMISSION_RATE = getRealModelParameter("SYMPTOMATIC_EMISSION_RATE", (63.5 / 60 / 60) * (1 / 13.8 * 60) * 1.0 * 1.8), // JLJ Table SI-4 (classroom / prepandemic scenario) = E_P0 * (1/breaths per hour) * r_E * (fe x fi)
	ASYMPTOMATIC_EMISSION_RATE = getRealModelParameter("ASYMPTOMATIC_EMISSION_RATE", SYMPTOMATIC_EMISSION_RATE), //same as symptomatic

	BREATHING_INHALATION_VOLUME = getRealModelParameter("BREATHING_INHALATION_VOLUME", 0.288 * (1 / 13.8 * 60) * 1.1 * 1.0); // JLJ Table SI-4 (classroom / prepandemic scenario) = B0 * (1/breaths per hour) * rB * fI

// Contact tracing parameters
parameter Real
	SYMPTOMATIC_DETECTION_PROB = getRealModelParameter("SYMPTOMATIC_DETECTION_PROB", 0.9),
	ASYMPTOMATIC_RANDOM_CONTACT_PROB = getRealModelParameter("ASYMPTOMATIC_RANDOM_CONTACT_PROB", 0.01);

// Grid constant
constant Integer
    VOLUMES_COUNT = GRID_DIVISIONS * GRID_DIVISIONS;


parameter Real
	INF = 1e20,
	EPS = 1e-5,
	PI = 3.1415926,
	GRID_SIZE = getRealModelParameter("GRID_SIZE", 20.0),
	CELL_EDGE_LENGTH = GRID_SIZE / GRID_DIVISIONS,
	Z_COORD = CELL_EDGE_LENGTH / 2.0;

/*
  Model variables
*/

// Particles position
Real x[N], y[N], z[N];

// Particles velocity
Real vx[N], vy[N], vz[N];

// Particles acceleration
Real ax[N], ay[N], az[N];

// Particles desired destination variables
discrete Real dx[N], dy[N], dz[N];

// Particles desired speed
discrete Real desiredSpeed[N];

// Volume concentration
Real volumeConcentration[VOLUMES_COUNT];

// Volume emission rate
Real volumeEmissionRate[VOLUMES_COUNT];

// Volume lambda
Real volumeLambda[VOLUMES_COUNT];

/*
  Time array variables used on triggering events with "when" statements
*/

// Useful variable for stopping the simulation when there is no more need to run
// Also it is used to set a maximum running time when developing or debugging
discrete Real terminateTime;

// Variables used to track statistics of infections, recoveries and isolations. These values are printed in the CSV output
discrete Integer infectedCount, recoveredCount, infectionsCount, isolatedCount;

// Variable used to control and trigger periodic output
discrete Real nextOutputTick;

// Variable used to control and trigger progress output in the terminal
discrete Real nextProgressTick;

// Variable used to control and trigger motivation update
discrete Real nextMotivationTick;

// Variable used to control destination update
discrete Real nextDestinationTick;

// This variable is used to simulate particles brething (they all breath at the same periodic time)
discrete Real nextBreathingTick;

// Variable used to control and trigger subway hub update
discrete Real nextSubwayHubTick;

// Disease evolution time-events variables
discrete Real expositionTime[N];
discrete Real infectionStartTime[N];
discrete Real infectionFinishTime[N];
discrete Real symptomsStartTime[N];


// local variables
discrete Real _, normalX, normalY, ux, uy, uz, hx, hy, hz, volumeID, groupID, desiredX, desiredY, desiredZ;
discrete Boolean isolate;


initial algorithm

	// sets the log level from the config file
    _ := setDebugLevel(getIntegerModelParameter("LOG_LEVEL", INFO()));
    _ := debug(INFO(), time, "Starting initial algorithm", _, _, _, _);

	// sets the random seed from the config file
	_ := random_reseed(RANDOM_SEED);

	// sets the parameters from the config file
	_ := setParameters();

	_ := setContagionConstants(SUSCEPTIBLE(), UNKNOWN(), PRE_SYMPTOMATIC(), PRESYMPTOMATIC_CONTAGION_PROB, SYMPTOMATIC(),
		SYMPTOMATIC_CONTAGION_PROB, ASYMPTOMATIC(), ASYMPTOMATIC_CONTAGION_PROB, PARTICLE_TO_PARTICLE_INFECTION_PROB);

    _ := debug(INFO(), time, "Grid setup. Divisions = %d", GRID_DIVISIONS, _, _, _);

	// setup the grid in RETQSS as a simple grid using the constants from config file
    _ := geometry_gridSetUp(GRID_DIVISIONS, GRID_DIVISIONS, 1, CELL_EDGE_LENGTH);

	for i in 1:VOLUMES_COUNT loop
		_ := volume_setProperty(i, "isObstacle", isInArrayParameter("OBSTACLES", i));
		_ := volume_setProperty(i, "isSubway", isInArrayParameter("SUBWAYS", i));
		volumeConcentration[i] := 0.0;
		volumeEmissionRate[i] := 0.0;
		volumeLambda[i] := VOLUME_CONCENTRATION_LAMBDA;
		_ := volume_setProperty(i, "particleSpeed", DEFAULT_SPEED);
		_ := volume_setProperty(i, "particleEmissionRate", SYMPTOMATIC_EMISSION_RATE);
		_ := volume_setProperty(i, "particleBreathingInhalationVolume", BREATHING_INHALATION_VOLUME);
		_ := volume_setProperty(i, "isClosedSpace", 1);
		_ := volume_setProperty(i, "isBlock", 0);
    end for;

	// setup the particles in RETQSS
    _ := debug(INFO(), time, "Particles setup. N = %d", N,_,_,_);	
	_ := setUpParticles(N, CELL_EDGE_LENGTH, GRID_DIVISIONS, x);
    _ := debug(INFO(), time, "Particles setup ended. N = %d", N,_,_,_);

	for i in 1:N loop
		_ := particle_setProperty(i, "isObjective", 1);
	end for;

	_ := setUpParticleMovement(N);

	// setup the particles half in the left side and half in the right side of the grid
	for i in 1:N loop
		_ := particle_setProperty(i, "isObjective", 1);
        (x[i], y[i], z[i], dx[i], dy[i], dz[i]) := getInitialPosition(i);
		desiredSpeed[i] := random_normal(SPEED_MU, SPEED_SIGMA);
		_ := particle_relocate(i, x[i], y[i], z[i], vx[i], vy[i], vz[i]);
		_ := debug(INFO(), time, "Particle %d setup ended", i,_,_,_);
    end for;

	// setup the walls in RETQSS
	_ := setUpWalls();


	_ := debug(INFO(), time, "Walls setup ended",_,_,_,_);

	// setup the particles initial state
	for i in 1:N loop
		_ := particle_setProperty(i, "initialX", x[i]);
		_ := particle_setProperty(i, "initialY", y[i]);
		_ := particle_setProperty(i, "initialVX", vx[i]);
		_ := particle_setProperty(i, "initialVY", vy[i]);
		if i <= INITIAL_EXPOSED then
			expositionTime[i] := EPS;
		else
			expositionTime[i] := INF;
		end if;
		if i <= INITIAL_INFECTED then
			infectionStartTime[i] := EPS;
			_ := particle_setProperty(i, "status", ASYMPTOMATIC());
			infectedCount := infectedCount + 1;
		else
			infectionStartTime[i] := INF;
			_ := particle_setProperty(i, "status", SUSCEPTIBLE());
		end if;
		symptomsStartTime[i] := INF;
		infectionFinishTime[i] := INF;
		_ := particle_setProperty(i, "enteredVolumesCount", 0.0);
		_ := particle_setProperty(i, "bouncesCount", 0.0);
		_ := particle_setProperty(i, "volumeCrossingLastId", particle_currentVolumeID(i));
        _ := particle_setProperty(i, "volumeCrossingTime", 0.0);
        _ := particle_setProperty(i, "volumeCrossingConcentration", 0.0);
		_ := particle_setProperty(i, "trackingStatus", UNKNOWN());
        _ := particle_setProperty(i, "infectionsCount", 0);
		_ := setRandomDiseaseOutcomesProperties(i, YOUNG_PROB, ADULT_PROB,
			YOUNG_SYMPTOMATIC_PROB, ADULT_SYMPTOMATIC_PROB, OLD_SYMPTOMATIC_PROB,
			YOUNG_DEATH_PROB,ADULT_DEATH_PROB,OLD_DEATH_PROB,
			INCUBATION_TIME_M, INCUBATION_TIME_S, PRESYMPTOMATIC_TIME_M, PRESYMPTOMATIC_TIME_S, SYMPTOMATIC_TIME_M, SYMPTOMATIC_TIME_S,
			SYMPTOMATIC_EMISSION_RATE, ASYMPTOMATIC_EMISSION_RATE, BREATHING_INHALATION_VOLUME);
    end for;

	// initialize statistics tracking variables
    recoveredCount := 0;
    infectionsCount := 0;
    isolatedCount := 0;

	if BORDER_IMPLEMENTATION == 3 then
		// update the neighboring volumes for each particle
		for i in 1:N loop
			_ := updateNeighboringVolumes(i, GRID_DIVISIONS);
		end for;
	end if;

    terminateTime := FORCE_TERMINATION_AT;
    nextProgressTick := EPS;
	nextMotivationTick := EPS;
	nextOutputTick := EPS;
	nextDestinationTick := DESTINATION_UPDATE_DT;
	nextBreathingTick := EPS;
    _ := debug(INFO(), time, "Done initial algorithm",_,_,_,_);
    _ := debug(INFO(), time, "Pedestrian implementation: %d", PEDESTRIAN_IMPLEMENTATION,_,_,_);
	_ := debug(INFO(), time, "Border implementation: %d", BORDER_IMPLEMENTATION,_,_,_);

    
/*
  Model's diferential equations: for particles movements and volumes concentration
*/
equation
    // newtonian position/velocity equations for each particle
    for i in 1:N loop
        der(x[i])  = vx[i];
        der(y[i])  = vy[i];
        der(z[i])  = vz[i];
        der(vx[i]) = ax[i];
        der(vy[i]) = ay[i];
        der(vz[i]) = az[i];
		der(ax[i]) = 0.0;
		der(ay[i]) = 0.0;
		der(az[i]) = 0.0;
    end for;

    // box model concentration equations for each volume
    for i in 1:VOLUMES_COUNT loop
        der(volumeConcentration[i]) = - volumeConcentration[i] * volumeLambda[i];
		der(volumeEmissionRate[i]) = 0.0;
		der(volumeLambda[i]) = 0.0;
	end for;

/*
  Model's time events
*/
algorithm	

	//EVENT: particle enters a volume and update neighboring volumes that are obstacles
	for i in 1:N loop
		when time > particle_nextCrossingTime(i,x[i],y[i],z[i],vx[i],vy[i],vz[i]) then
			(expositionTime, _, _) := onNextCross(time, i, CLOSE_CONTACT_PROB);
			if BORDER_IMPLEMENTATION == 3 then
				_ := updateNeighboringVolumes(i, GRID_DIVISIONS);
			end if;
		end when;

		//EVENT: particle is exposed
		when time > expositionTime[i] then
			infectedCount := infectedCount + 1;
			(infectionStartTime[i], _) := onExposition(time, i);
		end when;

		// EVENT: particle infection start
		when time > infectionStartTime[i] then
			(infectionFinishTime[i], symptomsStartTime[i], _) := onInfectionStart(time, i, ASYMPTOMATIC_RANDOM_CONTACT_PROB);
		end when;

		//EVENT: particle symptoms start (it may trigger the particle stop moving if the isolation starts)
		when time > symptomsStartTime[i] then
			(infectionFinishTime[i], _, _) := onSymptomsStart(time, i, SYMPTOMATIC_DETECTION_PROB, 0.0, 0.0, 0.0, 0.0);
		end when;

		//EVENT: particle infection finished (it may trigger the particle stop moving if died)
		when time > infectionFinishTime[i] then
			_ := onInfectionEnd(time, i);
			recoveredCount := recoveredCount + 1;
			infectionsCount := infectionsCount + particle_getProperty(i, "infectionsCount");
			if not shouldMove(i) then
				reinit(vx[i], 0.);
				reinit(vy[i], 0.);
				_ := particle_relocate(i, x[i], y[i], z[i], vx[i], vy[i], vz[i]);
			end if;
		end when;
	end for;


	//EVENT: all particles breath. New emissions rate are computed and particles may become infected. 
	when time > nextBreathingTick then
		nextBreathingTick := time + BREATHING_INTERVAL_TIME;
		for i in 1:VOLUMES_COUNT loop
			_ := volume_setProperty(i, "newVolumeEmissionRate", 0);
		end for;
		for i in 1:N loop
			if onBreathe(time, i, volumeConcentration) then
				expositionTime[i] := time + EPS;
			end if;
		end for;
		for i in 1:VOLUMES_COUNT loop
			reinit(volumeConcentration[i], volume_getProperty(i, "newVolumeEmissionRate")*BREATHING_INTERVAL_TIME+volumeConcentration[i]);
		end for;
	end when;


	//EVENT: Next CSV output time: prints a new csv line and computes the next output time incrementing the variable
	when time > nextOutputTick then
		_ := covid19_outputCSV(time, N, x, y, VOLUMES_COUNT, volumeConcentration, recoveredCount, infectedCount);
		nextOutputTick := time + OUTPUT_UPDATE_DT;
		if recoveredCount == infectedCount and terminateTime == 0 then
			terminateTime := time + nextOutputTick;
		end if;
	end when;

	//EVENT: Terminate time is reached, calling native function terminate()
	when time > terminateTime then
		terminate();
	end when;

	//EVENT: Next subway hub time: updates the particles destination
	when time > nextSubwayHubTick then
		nextSubwayHubTick := time + OBJECTIVE_SUBWAY_HUB_DT;
		for i in 1:N loop
			(dx[i], dy[i], dz[i]) := getNextPosition(i);
		end for;
	end when;
	
	when time > nextMotivationTick then
		nextMotivationTick := time + MOTIVATION_UPDATE_DT;
		for i in 1:N loop
			(dx[i], dy[i], dz[i]) := moveOutOfSubway(i, dx[i], dy[i], dz[i]);
		end for;

		if SOCIAL_FORCE_MODEL == 1 then
			// _ := debug(INFO(), time, "Updating particles motivation",_,_,_,_);
			for i in 1:N loop
				hx := dx[i];
				hy := dy[i];
				hz := dz[i];
				(hx, hy, hz) := pedestrianTotalMotivation(i, desiredSpeed, x, y, z, vx, vy, vz, hx, hy, hz, CELL_EDGE_LENGTH, VOLUMES_COUNT, N);
				reinit(ax[i], hx);
				reinit(ay[i], hy);
				reinit(az[i], hz);	
			end for;
		end if;
		if SOCIAL_FORCE_MODEL == 0 then
			_ := debug(INFO(), time, "Updating particles fixed velocity",_,_,_,_);
			// The desired direction is given by the difference between the current position and the target position
			for i in 1:N loop
				hx := dx[i];
				hy := dy[i];
				hz := dz[i];
				desiredX := dx[i] - x[i];
				desiredY := dy[i] - y[i];
				desiredZ := dz[i] - z[i];

				(desiredX, desiredY, desiredZ) := vectorWithNorm(
					desiredX, desiredY, desiredZ,
					1.3*SPEED_MU
				);

				// The desired acceleration is the difference between the desired speed and the current speed
				reinit(vx[i], desiredX);
				reinit(vy[i], desiredY);
				reinit(vz[i], desiredZ);
				reinit(ax[i], 0.0);
				reinit(ay[i], 0.0);
				reinit(az[i], 0.0);
			end for;
		end if;
	end when;

	//EVENT: Next progress output time: prints a new line in stdout and computes the next output time incrementing the variable
	when time > nextProgressTick then
		// _ := debug(INFO(), time, "Progress checkpoint",_,_,_,_);
        nextProgressTick := time + PROGRESS_UPDATE_DT;
	end when;
	

annotation(
	experiment(
		MMO_Description="Indirect infection of particles interacting through volumes.",
		MMO_Solver=QSS2,
		MMO_SymDiff=false,
		MMO_PartitionMethod=Metis,
		MMO_Scheduler=ST_Binary,
		Jacobian=Dense,
		StartTime=0.0,
		StopTime=10000.0,
		Tolerance={1e-5},
		AbsTolerance={1e-8}
	));

end subway_hub;
