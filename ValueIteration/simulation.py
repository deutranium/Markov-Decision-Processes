import json
import numpy as np



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


# MOVEMENT FUNCTIONS

def move_up(state):
    new_state = state.copy()
    
    # get the coordinates from the state
    old_coordinates = coordinates[direction_tuple[new_state[0]]]
    (x, y) = old_coordinates
    y += 1    # increment one direction along y-axis
    coord = (x, y)
    new_state[0] = direction_tuple.index(positions[coord]) # find the updated location as 0, 1, 2, 3 etc.
    
    return new_state
    

def move_down(state):
    new_state = state.copy()
    
    old_coordinates = coordinates[direction_tuple[new_state[0]]]
    (x, y) = old_coordinates
    y -= 1
    coord = (x, y)
    new_state[0] = direction_tuple.index(positions[coord])
    return new_state
    

def move_left(state):
    new_state = state.copy()
    
    old_coordinates = coordinates[direction_tuple[new_state[0]]]
    (x, y) = old_coordinates
    x -= 1
    coord = (x, y)
    new_state[0] = direction_tuple.index(positions[coord])
    return new_state
    

def move_right(state):
    new_state = state.copy()
    
    old_coordinates = coordinates[direction_tuple[new_state[0]]]
    (x, y) = old_coordinates
    x += 1
    coord = (x, y)
    new_state[0] = direction_tuple.index(positions[coord])
    return new_state
    

def move_stay(state):
    new_state = state.copy()
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






with open('best_actions.json') as f:
    policy = np.array(json.load(f))


def calc_prob(state, action):
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


initial_state = [1,0 ,0, 1, 4]


def simulate(state):
    # print(policy[tuple(state)])
    probs = calc_prob(state, policy[tuple(state)])

    this_states = np.array(probs[1])
    this_probs = np.array(probs[0])

    new_state = this_states[np.random.choice(np.arange(0, len(this_states)), p = this_probs)]

    return list(new_state)



while initial_state[4] != 0:
    trace_state = initial_state.copy()
    trace_state[0] = codes_loc[trace_state[0]]
    trace_state[3] = codes_state[trace_state[3]]
    trace_state[4] = trace_state[4]*25



    print(trace_state, ": ", policy[tuple(initial_state)])
    initial_state = simulate(initial_state.copy())
    # print(initial_state)
    # print()

print(initial_state, ": ", policy[tuple(initial_state)])
