package retQSS_social_force_model_params

import retQSS_utils;
import retQSS_social_force_model_utils;

// Pedestrian destination
// 0 - same Y coordinate
// 1 - opposite Y coordinate
// 2 - random Y coordinate
function PEDESTRIAN_DESTINATION
	output Real destination;
algorithm
	destination := getRealModelParameter("PEDESTRIAN_DESTINATION", 0);
end PEDESTRIAN_DESTINATION;

function PEDESTRIAN_IMPLEMENTATION
	output Integer implementation;
algorithm
	implementation := getIntegerModelParameter("PEDESTRIAN_IMPLEMENTATION", 0);
end PEDESTRIAN_IMPLEMENTATION;

function BORDER_IMPLEMENTATION
	output Integer implementation;
algorithm
	implementation := getIntegerModelParameter("BORDER_IMPLEMENTATION", 0);
end BORDER_IMPLEMENTATION;

function PEDESTRIAN_A_1
	output Real pedestrianA_1;
algorithm
	pedestrianA_1 := getRealModelParameter("PEDESTRIAN_A_1", 2.1);
end PEDESTRIAN_A_1;

function PEDESTRIAN_B_1
	output Real pedestrianB_1;
algorithm
	pedestrianB_1 := getRealModelParameter("PEDESTRIAN_B_1", 0.3);
end PEDESTRIAN_B_1;

function PEDESTRIAN_A_2
	output Real pedestrianA_2;
algorithm
	pedestrianA_2 := getRealModelParameter("PEDESTRIAN_A_2", 2.1);
end PEDESTRIAN_A_2;

function PEDESTRIAN_B_2
	output Real pedestrianB_2;
algorithm
	pedestrianB_2 := getRealModelParameter("PEDESTRIAN_B_2", 0.3);
end PEDESTRIAN_B_2;

function PEDESTRIAN_R
	output Real pedestrianR;
algorithm
	pedestrianR := getRealModelParameter("PEDESTRIAN_R", 0.1);
end PEDESTRIAN_R;

function PEDESTRIAN_LAMBDA
	output Real pedestrianLambda;
algorithm
	pedestrianLambda := getRealModelParameter("PEDESTRIAN_LAMBDA", 0.3);
end PEDESTRIAN_LAMBDA;

function RELAXATION_TIME
	output Real relaxationTime;
algorithm
	relaxationTime := getRealModelParameter("RELAXATION_TIME", 0.5);
end RELAXATION_TIME;

function BORDER_A
	output Real borderA;
algorithm
	borderA := getRealModelParameter("BORDER_A", 10.0);
end BORDER_A;

function BORDER_B
	output Real borderB;
algorithm
	borderB := getRealModelParameter("BORDER_B", 0.7);
end BORDER_B;

function BORDER_R
	output Real borderR;
algorithm
	borderR := getRealModelParameter("BORDER_R", 0.1);
end BORDER_R;

function FROM_Y
	output Real fromY;
algorithm
	fromY := getRealModelParameter("FROM_Y", 0.0);
end FROM_Y;

function TO_Y
	output Real toY;
algorithm
	toY := getRealModelParameter("TO_Y", 20.0);
end TO_Y;

function GRID_SIZE
	output Real gridSize;
algorithm
	gridSize := getRealModelParameter("GRID_SIZE", 20.0);
end GRID_SIZE;

end retQSS_social_force_model_params;