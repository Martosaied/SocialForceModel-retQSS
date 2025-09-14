model helbing_not_qss

import retQSS;
import retQSS_utils;
import retQSS_social_force_model_utils;
import retQSS_social_force_model_params;
import retQSS_social_force_model_types;
import retQSS_helbing_not_qss;

/*
  This section loads parameters and constants from the parameters.config file
*/

constant Integer
	N = 100;

// Initial conditions parameters
parameter Integer
	RANDOM_SEED = getIntegerModelParameter("RANDOM_SEED", 0),
	FORCE_TERMINATION_AT = getRealModelParameter("FORCE_TERMINATION_AT", 40),
	CONVEYOR_BELT_EFFECT = getIntegerModelParameter("CONVEYOR_BELT_EFFECT", 0);

// Output delta time parameter
parameter Real
	DEFAULT_SPEED = getRealModelParameter("DEFAULT_SPEED", 1.34),
	OUTPUT_UPDATE_DT = getRealModelParameter("OUTPUT_UPDATE_DT", 0.1),
	SPEED_MU = getRealModelParameter("SPEED_MU", 1.34),
	SPEED_SIGMA = getRealModelParameter("SPEED_SIGMA", 0.26),
	FROM_Y = getRealModelParameter("FROM_Y", 0.0),
	TO_Y = getRealModelParameter("TO_Y", 20.0);


parameter Real
	INF = 1e20,
	EPS = 1e-5,
	PI = 3.1415926,
	PROGRESS_UPDATE_DT = getRealModelParameter("PROGRESS_UPDATE_DT", 0.1),
    MOTIVATION_UPDATE_DT = getRealModelParameter("MOTIVATION_UPDATE_DT", 0.1),
	GRID_SIZE = getRealModelParameter("GRID_SIZE", 20.0);

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

// Particles group IDs
discrete Integer groupIDs[N];


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
discrete Real _, ux, uy, uz, hx, hy, hz;
discrete Integer groupID;


initial algorithm

	// sets the log level from the config file
    _ := setDebugLevel(getIntegerModelParameter("LOG_LEVEL", INFO()));
    _ := debug(INFO(), time, "Starting initial algorithm", _, _, _, _);

	// sets the random seed from the config file
	_ := random_reseed(RANDOM_SEED);

	// sets the parameters from the config file
	_ := setParameters();

	// setup the particles half in the left side and half in the right side of the grid
	for i in 1:N loop
        (groupID, x[i], y[i], z[i], dx[i], dy[i], dz[i]) := randomRoute(GRID_SIZE, FROM_Y, TO_Y);
		desiredSpeed[i] := random_normal(SPEED_MU, SPEED_SIGMA);
		groupIDs[i] := groupID;
    end for;

    terminateTime := FORCE_TERMINATION_AT;
    nextProgressTick := EPS;
	nextMotivationTick := EPS;
	nextOutputTick := EPS;
    _ := debug(INFO(), time, "Done initial algorithm",_,_,_,_);

    
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

/*
  Model's time events
*/
algorithm	

	//EVENT: Next CSV output time: prints a new csv line and computes the next output time incrementing the variable
	when time > nextOutputTick then
		_ := debug(INFO(), time, "Updating particles output",_,_,_,_);
		_ := outputCSV(time, N, groupIDs, x, y, vx, vy);
		nextOutputTick := time + OUTPUT_UPDATE_DT;
	end when;

	//EVENT: Terminate time is reached, calling native function terminate()
	when time > terminateTime then
		terminate();
	end when;
	
	//EVENT: Next motivation update time: updates the particles motivation and computes the next motivation update time incrementing the variable
	when time > nextMotivationTick then
		nextMotivationTick := time + MOTIVATION_UPDATE_DT;
		_ := debug(INFO(), time, "Updating particles motivation",_,_,_,_);
		for i in 1:N loop
			hx := dx[i];
			hy := dy[i];
			hz := dz[i];
			(hx, hy, hz) := pedestrianTotalMotivation(i, N, x, y, z, vx, vy, vz, hx, hy, hz);
			reinit(ax[i], hx);
			reinit(ay[i], hy);
			reinit(az[i], hz);	
		end for;

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
				end if;
				if hy <> y[i] then
					reinit(y[i], hy);
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

end helbing_not_qss;
