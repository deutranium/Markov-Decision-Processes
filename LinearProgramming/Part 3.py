import cvxpy as cp
import numpy as np

############################################################################
###############################  PARAMETERS  ###############################
############################################################################

STEPCOST = -5
POSITIONS = 5
MAX_MATERIALS = 2         # num_material = 0..2
MAX_ARROWS = 3            # num_arrow = 0..3
MONSTER_STATES = 2
MAX_HEALTH_STATES = 4     # health = (0..4)*25

############################################################################
###########  Variables to keep track of states, state-action...  ###########
############################################################################
state_to_idx = np.zeros((POSITIONS, MAX_MATERIALS+1, MAX_ARROWS+1, MONSTER_STATES, MAX_HEALTH_STATES+1))
states = []
state_action_pairs = []

############################################################################
##########################  Probablity Function  ###########################
############################################################################

# MOVEMENT FUNCTIONS

def move_up(state):
    new_state = list(state.copy())
    
    # get the coordinates from the state
    old_coordinates = coordinates[direction_tuple[new_state[0]]]
    (x, y) = old_coordinates
    y += 1    # increment one direction along y-axis
    coord = (x, y)
    new_state[0] = direction_tuple.index(positions[coord]) # find the updated location as 0, 1, 2, 3 etc.
    
    return new_state
    

def move_down(state):
    new_state = list(state.copy())
    
    old_coordinates = coordinates[direction_tuple[new_state[0]]]
    (x, y) = old_coordinates
    y -= 1
    coord = (x, y)
    new_state[0] = direction_tuple.index(positions[coord])
    return new_state
    

def move_left(state):
    new_state = list(state.copy())
    
    old_coordinates = coordinates[direction_tuple[new_state[0]]]
    (x, y) = old_coordinates
    x -= 1
    coord = (x, y)
    new_state[0] = direction_tuple.index(positions[coord])
    return new_state
    

def move_right(state):
    new_state = list(state.copy())
    
    old_coordinates = coordinates[direction_tuple[new_state[0]]]
    (x, y) = old_coordinates
    x += 1
    coord = (x, y)
    new_state[0] = direction_tuple.index(positions[coord])
    return new_state
    

def move_stay(state):
    new_state = list(state.copy())
    return new_state
    

def move(action, state):
    switcher = {
        "UP": move_up,
        "DOWN": move_down,
        "LEFT": move_left,
        "RIGHT": move_right,
        "STAY": move_stay
    }
    
    return(switcher[action](state))


# directional CONSTANTS and UTILITIES

positions = {
    (0,0): "CENTER",
    (-1,0): "WEST",
    (1,0): "EAST",
    (0,1): "NORTH",
    (0,-1): "SOUTH"
}

coordinates = {
    "CENTER": (0,0),
    "WEST": (-1,0),
    "EAST": (1,0),
    "NORTH": (0,1),
    "SOUTH": (0,-1)
}

# actions possible at each location
possible_actions = {
    "CENTER": ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "SHOOT", "HIT"],
    "WEST": ["RIGHT", "STAY", "SHOOT"],
    "EAST": ["LEFT", "STAY", "SHOOT", "HIT"],
    "NORTH": ["DOWN", "STAY", "CRAFT"],
    "SOUTH": ["UP", "STAY", "GATHER"]
}

# eg. the first element of a state tuple is 1 -> state corresponds to direction_tuple[1] = "WEST" location
direction_tuple = ["CENTER", "WEST", "EAST", "NORTH", "SOUTH"]
codes_loc = ["C", "W", "E", "N", "S"]
codes_state = ["D", "R"]

def calc_prob(state, action):
#     print(state[0])
    location = direction_tuple[state[0]]
    
    ret = []
    
    if location == "NORTH":
        if action in ["DOWN", "STAY"]:
            new_state = move(action, state.copy())
            fail_state = state.copy()
            fail_state[0] = 2
            ret = [[0.85, 0.15], [new_state, fail_state]]
        else:
            arrow_probs = [0.5, 0.35, 0.15]
            arrow1 = state.copy()
            arrow2 = state.copy()
            arrow3 = state.copy()
            arrow1[1] -= 1
            arrow2[1] -= 1
            arrow3[1] -= 1
            arrow1[2] = min(3, arrow1[2] + 1)
            arrow2[2] = min(3, arrow2[2] + 2)
            arrow3[2] = min(3, arrow3[2] + 3)
            
            ret = [arrow_probs, [arrow1, arrow2, arrow3]]
            
  
    elif location == "SOUTH":
        if action in ["UP", "STAY"]:
            new_state = move(action, state.copy())
            fail_state = state.copy()
            fail_state[0] = 2
            ret = [[0.85, 0.15], [new_state, fail_state]]
        else:
            gather_probs = [0.75, 0.25]
            new_state = state.copy()
            fail_state = new_state.copy()
            new_state[1] = min(2, new_state[1] + 1)
            ret = [gather_probs, [new_state, fail_state]]
                
                
    elif location == "WEST":
        if action in ["RIGHT", "STAY"]:
            new_state = move(action, state.copy())
            fail_state = state.copy()
            fail_state[0] = 2
            ret = [[1, 0], [new_state, fail_state]]
        else:
            shoot_probs = [0.25, 0.75]
            new_state = state.copy()
            new_state[2] -= 1
            fail_state = new_state.copy()
            new_state[4] -= 1
            ret = [shoot_probs, [new_state, fail_state]]
            
            
    elif location == "EAST":
        if action in ["LEFT", "STAY"]:
            new_state = move(action, state.copy())
            fail_state = state.copy()
            fail_state[0] = 2
            ret = [[1, 0], [new_state, fail_state]]
        elif action == "SHOOT":
            shoot_probs = [0.9, 0.1]
            new_state = state.copy()
            new_state[2] -= 1
            fail_state = new_state.copy()
            new_state[4] -= 1
            ret = [shoot_probs, [new_state, fail_state]]
        else:
            hit_probs = [0.2, 0.8]
            new_state = state.copy()
            fail_state = new_state.copy()
            new_state[4] = max(0, new_state[4] - 2)
            ret = [hit_probs, [new_state, fail_state]]
            
            
    elif location == "CENTER":
        if action in ["UP", "DOWN", "RIGHT", "LEFT", "STAY"]:
            new_state = move(action, state.copy())
            fail_state = state.copy()
            fail_state[0] = 2
            ret = [[0.85, 0.15], [new_state, fail_state]]
        elif action == "SHOOT":
            shoot_probs = [0.5, 0.5]
            new_state = state.copy()
            new_state[2] -= 1
            fail_state = new_state.copy()
            new_state[4] -= 1
            ret = [shoot_probs, [new_state, fail_state]]
        else:
            hit_probs = [0.1, 0.9]
            new_state = state.copy()
            fail_state = new_state.copy()
            new_state[4] = max(0, new_state[4] - 2)
            ret = [hit_probs, [new_state, fail_state]]
            
    # monster slep slep 
    if not state[3]:
        awake_prob = [0.2, 0.8]
        probabilities = ret[0]
        states = ret[1]
        
        new_probs = []
        new_states = []
        
        for i, prob in enumerate(probabilities):
            new_probs.append(prob*awake_prob[0])
            new_probs.append(prob*awake_prob[1])
            awake_state = states[i].copy()
            awake_state[3] = 1
            new_states.append(awake_state)
            new_states.append(states[i].copy())
        
        ret = [new_probs, new_states]

    # monster wakey wakey   
    else:
        probabilities = ret[0]
        states = ret[1]
        
        new_probs = []
        new_states = []
        
        if location == "EAST" or "CENTER":
            attacc_prob = [0.5, 0.5]
            
            for i, prob in enumerate(probabilities):
                new_probs.append(prob*attacc_prob[0])
                new_states.append(states[i].copy())
            new_probs.append(attacc_prob[1])
            
            og_state = state.copy()
            og_state[2] = 0 # arrows
            og_state[4] = min(4, og_state[4] + 1) # helth
            og_state[3] = 0 # dormant state
            new_states.append(og_state)
            
        else:
            awake_prob = [0.5, 0.5]
            probabilities = ret[0]
            states = ret[1]

            for i, prob in enumerate(probabilities):
                new_probs.append(prob*awake_prob[0])
                new_probs.append(prob*awake_prob[1])
                awake_state = states[i].copy()
                awake_state[3] = 0
                new_states.append(awake_state)
                new_states.append(states[i].copy())
        ret = [new_probs, new_states]
            
    return ret

############################################################################
####################  Functions to define params for LP  ###################
############################################################################

def Reward(prev, curr):
    if ((prev[3] == 1) and(curr[3]==0)):
        if ((prev[0] in [0,2]) and (prev[4]>0)):
            return -40
    return 0


def initialize():
    temp = []
    count = 0
    for state_x, _ in np.ndenumerate(state_to_idx):
        state_to_idx[state_x] = count
        states.append(state_x)

        actions = possible_actions[direction_tuple[state_x[0]]].copy()        

        if "SHOOT" in actions:
            if state_x[2] == 0:
                actions.remove("SHOOT")
        if "CRAFT" in actions:
            if state_x[1] == 0:
                actions.remove("CRAFT")
        
        if state_x[4] == 0:
            actions = ["NONE"]
        
        for flem in actions:
            temp.append((state_x, flem))
        count += 1
    return temp


def calculate_A():
    final = np.zeros((len(states),len(state_action_pairs)))

    for i, state_action_x in enumerate(state_action_pairs):
        if state_action_x[1] != "NONE":
            prob, state_x = calc_prob(list(state_action_x[0]), state_action_x[1])
            final[int(state_to_idx[tuple(state_action_x[0])])][i] += np.sum(prob)
            for j, p in enumerate(prob):
                final[int(state_to_idx[tuple(state_x[j])])][i] -= p
        else:
            final[int(state_to_idx[state_action_x[0]])][i] = 1

    return final


def calculate_R():
    final = np.zeros(len(state_action_pairs))

    for i , state_action in enumerate(state_action_pairs):
        if state_action[1] != "NONE":
            probablity, state = calc_prob(list(state_action[0]), state_action[1])
            for j, prob in enumerate(probablity):
                final[i] += prob*( Reward(state_action[0], state[j]) + STEPCOST )
    return final

############################################################################
###########################  Linear Programming  ###########################
############################################################################

def LinProg(A, alpha, r):
    x = cp.Variable(shape=(len(state_action_pairs),1), name="x")
    
    constraints = [cp.matmul(A, x) == alpha, x >= 0]
    objective = cp.Maximize(cp.matmul(r,x))
    problem = cp.Problem(objective, constraints)
    
    return x, problem.solve()


# ["C", "W", "E", "N", "S"]
state_action_pairs = initialize()
A = calculate_A()
R = calculate_R()
alpha = np.full((len(states),1),1/600)

sol = LinProg(A, alpha, R)

############################################################################
#################################  Policy  #################################
############################################################################

def Generate_Policy():
    global sol, state_action_pairs, states
    output = []
    for i in states:
        max_reward, action = -np.inf, ""
        for j, pair in enumerate(state_action_pairs):
            if pair[0] == i:
                if max_reward < sol[0].value[j][0]:
                    max_reward = sol[0].value[j][0]
                    action = pair[1]
        temp = list(i).copy()
        temp[0], temp[3], temp[4] = codes_loc[i[0]], codes_state[i[3]], temp[4]*25
        output.append(([tuple(temp),action]))
    return output

############################################################################
#################################  Output  #################################
############################################################################

import json

output_dict = dict()
output_dict["a"] = [ list(i) for i in A ]
output_dict["r"] = list(R)
output_dict["alpha"] = [ i[0] for i in alpha]
output_dict["x"] = [i[0] for i in sol[0].value]
output_dict["policy"] = Generate_Policy()
output_dict["objective"] = sol[1]

out_file = open("./outputs/part_3_output.json", 'w')
json.dump(output_dict, out_file)
out_file.close()

for i in output_dict:
	if i !="objective":
		print("shape of", i, ":")
		print(np.array(output_dict[i], dtype=object).shape)
	else:
		print( i, ":")
		print(output_dict[i])
	