model helbing_equation

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
	N = 300;

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

// Helbing model parameters (simplified - only first term)
parameter Real
	PEDESTRIAN_A1 = getRealModelParameter("PEDESTRIAN_A1", 2000.0),
	PEDESTRIAN_B1 = getRealModelParameter("PEDESTRIAN_B1", 0.08),
	PEDESTRIAN_R = getRealModelParameter("PEDESTRIAN_R", 0.2),
	RELAXATION_TIME = getRealModelParameter("RELAXATION_TIME", 0.5);

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

// Desired velocities for Helbing model
Real desiredVx[N], desiredVy[N], desiredVz[N];

// Repulsive forces from other pedestrians
Real repulsiveFx[N], repulsiveFy[N], repulsiveFz[N];


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
  Model's diferential equations: Helbing social force model
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
    
    // Helbing social force model equations
	for i in 1:N loop
        // Desired velocity calculation
        desiredVx[i] = (dx[i] - x[i]) / sqrt((dx[i] - x[i])^2 + (dy[i] - y[i])^2) * desiredSpeed[i];
        desiredVy[i] = (dy[i] - y[i]) / sqrt((dx[i] - x[i])^2 + (dy[i] - y[i])^2) * desiredSpeed[i];
        desiredVz[i] = 0.0;
        
        // Repulsive forces from other pedestrians
		for j in 1:N loop
			auxRepulsiveFx[] = PEDESTRIAN_A1 * exp((2 * PEDESTRIAN_R - sqrt((x[j] - x[i])^2 + (y[j] - y[i])^2)) / PEDESTRIAN_B1) * (x[i] - x[j]) / sqrt((x[j] - x[i])^2 + (y[j] - y[i])^2) * (1 - (i == j));
			auxRepulsiveFy[i] = PEDESTRIAN_A1 * exp((2 * PEDESTRIAN_R - sqrt((x[j] - x[i])^2 + (y[j] - y[i])^2)) / PEDESTRIAN_B1) * (y[i] - y[j]) / sqrt((x[j] - x[i])^2 + (y[j] - y[i])^2) * (1 - (i == j));
		end for;
        repulsiveFz[i] = 0.0;
        
        // Total acceleration from Helbing model (desired velocity + repulsive forces)
        ax[i] = (desiredVx[i] - vx[i]) / 0.5 + sum(auxRepulsiveFx[i]);
        ay[i] = (desiredVy[i] - vy[i]) / 0.5 + sum(auxRepulsiveFy[i]);
        az[i] = 0.0;
    end for;


/*
  Model's time events
*/
algorithm	

	//EVENT: Next CSV output time: prints a new csv line and computes the next output time incrementing the variable
	when time > nextOutputTick then
		_ := debug(INFO(), time, "Updating particles output",_,_,_,_);
		_ := outputCSV(time, N, x, y, vx, vy, groupIDs);
		nextOutputTick := time + OUTPUT_UPDATE_DT;
	end when;

	//EVENT: Terminate time is reached, calling native function terminate()
	when time > terminateTime then
		terminate();
	end when;
	
	//EVENT: Next motivation update time: updates the particles destination and computes the next motivation update time incrementing the variable
	when time > nextMotivationTick then
		nextMotivationTick := time + MOTIVATION_UPDATE_DT;
		_ := debug(INFO(), time, "Updating particles destination",_,_,_,_);
		// Conveyor belt effect for periodic boundaries
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
		MMO_Description="Helbing social force model for pedestrian dynamics with continuous equations.",
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

end helbing_equation;
