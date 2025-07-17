model social_force_model

import retQSS;
import retQSS_social_force_model;
import retQSS_utils;
import retQSS_social_force_model_utils;
import retQSS_social_force_model_params;
import retQSS_social_force_model_types;

/*
  This section loads parameters and constants from the parameters.config file
*/

constant Integer
	N = 300,
	GRID_DIVISIONS = 60,
	LEFT_COUNT = N / 2;

// Initial conditions parameters
parameter Integer
	RANDOM_SEED = getIntegerModelParameter("RANDOM_SEED", 0),
	FORCE_TERMINATION_AT = getRealModelParameter("FORCE_TERMINATION_AT", 40),
	PEDESTRIAN_IMPLEMENTATION = getIntegerModelParameter("PEDESTRIAN_IMPLEMENTATION", 0),
	BORDER_IMPLEMENTATION = getIntegerModelParameter("BORDER_IMPLEMENTATION", 0),
	CONVEYOR_BELT_EFFECT = getIntegerModelParameter("CONVEYOR_BELT_EFFECT", 0);

// Output delta time parameter
parameter Real
	DEFAULT_SPEED = getRealModelParameter("DEFAULT_SPEED", 1.34),
	OUTPUT_UPDATE_DT = getRealModelParameter("OUTPUT_UPDATE_DT", 0.1),
	SPEED_MU = getRealModelParameter("SPEED_MU", 1.34),
	SPEED_SIGMA = getRealModelParameter("SPEED_SIGMA", 0.26),
	FROM_Y = getRealModelParameter("FROM_Y", 0.0),
	TO_Y = getRealModelParameter("TO_Y", 20.0);


// Grid constant
constant Integer
    VOLUMES_COUNT = GRID_DIVISIONS * GRID_DIVISIONS;


parameter Real
	INF = 1e20,
	EPS = 1e-5,
	PI = 3.1415926,
	PROGRESS_UPDATE_DT = 0.01,
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

Real volumeConcentration[VOLUMES_COUNT];

/*
  Time array variables used on triggering events with "when" statements
*/

// Useful variable for stopping the simulation when there is no more need to run
// Also it is used to set a maximum running time when developing or debugging
discrete Real terminateTime;


// Variable used to control and trigger periodic output
discrete Real nextOutputTick;

// Variable used to control and trigger progress output in the terminal
discrete Real nextProgressTick;

// Variable used to control and trigger motivation update
discrete Real nextMotivationTick;

// local variables
discrete Real _, normalX, normalY, ux, uy, uz, hx, hy, hz, volumeID, groupID;
discrete Boolean isolate;


initial algorithm

	// sets the log level from the config file
    _ := setDebugLevel(getIntegerModelParameter("LOG_LEVEL", INFO()));
    _ := debug(INFO(), time, "Starting initial algorithm", _, _, _, _);

	// sets the random seed from the config file
	_ := random_reseed(RANDOM_SEED);

	// sets the parameters from the config file
	_ := setParameters();

    _ := debug(INFO(), time, "Grid setup. Divisions = %d", GRID_DIVISIONS, _, _, _);

	// setup the grid in RETQSS as a simple grid using the constants from config file
    _ := geometry_gridSetUp(GRID_DIVISIONS, GRID_DIVISIONS, 1, CELL_EDGE_LENGTH);

	for i in 1:VOLUMES_COUNT loop
		_ := volume_setProperty(i, "isObstacle", isInArrayParameter("OBSTACLES", i));
		volumeConcentration[i] := 0.0;
    end for;

	// setup the particles in RETQSS
    _ := debug(INFO(), time, "Particles setup. N = %d", N,_,_,_);	
	_ := setUpParticles(N, CELL_EDGE_LENGTH, GRID_DIVISIONS, x);
    _ := debug(INFO(), time, "Particles setup ended. N = %d", N,_,_,_);

	// setup the particles half in the left side and half in the right side of the grid
	for i in 1:N loop
        (groupID, x[i], y[i], z[i], dx[i], dy[i], dz[i]) := randomRoute(GRID_SIZE, Z_COORD, FROM_Y, TO_Y);
		_ := particle_setProperty(i, "type", groupID);
		desiredSpeed[i] := random_normal(SPEED_MU, SPEED_SIGMA);
		_ := particle_relocate(i, x[i], y[i], z[i], vx[i], vy[i], vz[i]);
    end for;

	// setup the walls in RETQSS
	_ := setUpWalls();


	// setup the particles initial state
	for i in 1:N loop
		_ := particle_setProperty(i, "initialX", x[i]);
		_ := particle_setProperty(i, "initialY", y[i]);
		_ := particle_setProperty(i, "initialVX", vx[i]);
		_ := particle_setProperty(i, "initialVY", vy[i]);
    end for;

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

	for i in 1:VOLUMES_COUNT loop
        der(volumeConcentration[i]) = 0.;
    end for;

/*
  Model's time events
*/
algorithm	

	//EVENT: particle enters a volume and update neighboring volumes that are obstacles
	for i in 1:N loop
		when time > particle_nextCrossingTime(i,x[i],y[i],z[i],vx[i],vy[i],vz[i]) then
			if BORDER_IMPLEMENTATION == 3 then
				_ := updateNeighboringVolumes(i, GRID_DIVISIONS);
			end if;
		end when;
	end for;

	//EVENT: Next CSV output time: prints a new csv line and computes the next output time incrementing the variable
	when time > nextOutputTick then
		_ := social_force_model_outputCSV(time, N, x, y, vx, vy);
		nextOutputTick := time + OUTPUT_UPDATE_DT;
	end when;

	//EVENT: Terminate time is reached, calling native function terminate()
	when time > terminateTime then
		terminate();
	end when;

	
	when time > nextMotivationTick then
		nextMotivationTick := time + PROGRESS_UPDATE_DT;
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

		_ := debug(INFO(), time, "Updating particles position",_,_,_,_);
		for i in 1:N loop
			hx := x[i];
			hy := y[i];
			if CONVEYOR_BELT_EFFECT == 1 then
				if y[i] < 0.0 then
					hy := GRID_SIZE;
				end if;
				if y[i] > GRID_SIZE then
					hy := 0.0;
				end if;
				if x[i] < 0.0 then
					hx := GRID_SIZE;
				end if;
				if x[i] > GRID_SIZE then
					hx := 0.0;
				end if;
				
				if hx <> x[i] then
					reinit(x[i], hx);
					_ := particle_relocate(i, hx, hy, z[i], vx[i], vy[i], vz[i]);
				end if;
			end if;
		end for;
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
		StopTime=1000.0,
		Tolerance={1e-5},
		AbsTolerance={1e-8}
	));

end social_force_model;
