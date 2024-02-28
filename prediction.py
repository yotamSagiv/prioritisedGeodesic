import numpy as np
import matplotlib.pyplot as plt

import time

from graph import DiGraph
from geodesic_agent import GeodesicAgent, SftmxGeodesicAgent
from RL_utils import softmax

np.random.seed(865612)


### Convenience function
def build_gdm(p_stay_low, p_stay_high=0.95):
    """
    Builds the goal distribution matrix. There is a low-occupancy state (0) and two high-occupancy states (1 and 2).
    State 0 has a self-transition with probability `p_stay_low` and a transitition to state 1 with probability
    `1 - p_stay_low`. States 1 and 2 each transition to state 0 with low probability and otherwise uniformly switch
    between themselves.

    Args:
        p_stay_low (float): Stay probability at the low occupancy goal.

    Returns:
        The goal distribution matrix where `gdm[t, s]` indicates P(next goal = t | current goal = s).
    """
    gdm = np.zeros((num_goals, num_goals))
    gdm[0, 0] = p_stay_low
    gdm[1, 0] = 1 - p_stay_low

    gdm[1, 1] = p_stay_high
    gdm[0, 1] = 1 - p_stay_high

    return gdm

def action_seq(goal, arm_length, start_to_choice):
   pass

### Main script


## Set parameters
# Geometry
num_goals = 2
num_bottlenecks = 3

dist_opt_bn = 3  # Distance to the single optimal bottleneck
dist_subopt_bns = 4  # Distance to the two suboptimal bottlenecks

dist_opt_goal = 3  # Distance from the optimal bottleneck to the goals
dist_subopt_near = 4  # Distance from the suboptimal bottleneck to its nearest set of goals
dist_subopt_far = 5  # Distance from the suboptimal bottleneck to its farthest set of goals

### Build underlying MDP
# Basics
num_states = 1 + num_goals + num_bottlenecks  # Distinguished states (start + bottlenecks + goals)
num_states += (dist_opt_bn - 1) + 2 * (dist_subopt_bns - 1)  # Distance from start to each bottleneck
num_states += num_goals * (dist_opt_goal - 1)  # Distance from optimal bottleneck to each goal
num_states += (dist_subopt_near - 1) + (dist_subopt_far - 1)  # Distance from one suboptimal bottleneck to its goals
num_states += (dist_subopt_far - 1) + (dist_subopt_near - 1)  # Distance from the other suboptimal bottleneck
num_actions = 5  # 0-2: make a choice, 3: proceed down arm, 4: reverse across arm

# Useful IDs
start_id = 0
bn_ids = list(1 + np.arange(num_bottlenecks))
goal_ids = list(1 + len(bn_ids) + np.arange(num_goals))
num_special_states = 1 + len(bn_ids) + len(goal_ids)

## Fill out edges
edges = np.zeros((num_states, num_actions))

# Set default action outcome to be null (overwrite later for meaningful actions)
for s in range(num_states):
    for a in range(num_actions):
        edges[s, a] = s

# Connect start state to bottleneck arm start states
bn_arm_start_ids = [num_special_states, num_special_states + (dist_opt_bn - 1),
                    num_special_states + (dist_opt_bn - 1) + (dist_subopt_bns - 1)]

for i in range(num_bottlenecks):
    arm_start_id = bn_arm_start_ids[i]
    edges[0, i] = arm_start_id
    edges[arm_start_id, 4] = 0  # Reverse action from arm back to start state

# Connect optimal bottleneck arm
arm_start_id = bn_arm_start_ids[0]
for j in range(dist_opt_bn - 1):
    # Forward actions
    if j != dist_opt_bn - 2:
        edges[arm_start_id + j, 3] = arm_start_id + j + 1
    else:
        edges[arm_start_id + j, 3] = bn_ids[0]

    # Backward actions
    if j != 0:
        edges[arm_start_id + j, 4] = arm_start_id + j - 1

# Connect suboptimal bottleneck arms
for i in range(1, num_bottlenecks):
    arm_start_id = bn_arm_start_ids[i]
    for j in range(dist_subopt_bns - 1):
        # Forward actions
        if j != dist_subopt_bns - 2:
            edges[arm_start_id + j, 3] = arm_start_id + j + 1
        else:
            edges[arm_start_id + j, 3] = bn_ids[i]

        # Backward actions
        if j != 0:
            edges[arm_start_id + j, 4] = arm_start_id + j - 1

# Connect each bottleneck to its associated goal arms
opt_start_id = num_special_states + (dist_opt_bn - 1) + (2 * (dist_subopt_bns - 1))

# Optimal bottleneck
for i in range(num_goals):
    arm_start_id = opt_start_id + (i * (dist_opt_goal - 1))
    edges[bn_ids[0], i] = arm_start_id
    edges[arm_start_id, 4] = bn_ids[0]

    for j in range(dist_opt_goal - 1):
        # Forward actions
        if j != dist_opt_goal - 2:
            edges[arm_start_id + j, 3] = arm_start_id + j + 1
        else:
            edges[arm_start_id + j, 3] = goal_ids[i]

        # Backward actions
        if j != 0:
            edges[arm_start_id + j, 4] = arm_start_id + j - 1

# Low-frequency bottleneck
lf_start_id = opt_start_id + num_goals * (dist_opt_goal - 1)
lf_distances = [dist_subopt_near - 1, dist_subopt_far - 1]
for i in range(num_goals):
    arm_start_id = int(lf_start_id + sum(lf_distances[:i]))
    edges[bn_ids[1], i] = arm_start_id
    edges[arm_start_id, 4] = bn_ids[1]

    for j in range(lf_distances[i]):
        # Forward actions
        if j != lf_distances[i] - 1:
            edges[arm_start_id + j, 3] = arm_start_id + j + 1
        else:
            edges[arm_start_id + j, 3] = goal_ids[i]

        # Backward actions
        if j != 0:
            edges[arm_start_id + j, 4] = arm_start_id + j - 1

# High-frequency bottleneck
hf_start_id = lf_start_id + sum(lf_distances)
hf_distances = [dist_subopt_far - 1, dist_subopt_near - 1]
for i in range(num_goals):
    arm_start_id = int(hf_start_id + sum(hf_distances[:i]))
    edges[bn_ids[2], i] = arm_start_id
    edges[arm_start_id, 4] = bn_ids[2]

    for j in range(hf_distances[i]):
        # Forward actions
        if j != hf_distances[i] - 1:
            edges[arm_start_id + j, 3] = arm_start_id + j + 1
        else:
            edges[arm_start_id + j, 3] = goal_ids[i]

        # Backward actions
        if j != 0:
            edges[arm_start_id + j, 4] = arm_start_id + j - 1

for i in range(num_states):
    print(i, edges[i, :])

# Build edges variant with path to optimal bottleneck sealed
wall_edges = edges.copy()
wall_loc = bn_arm_start_ids[0] + 1
wall_edges[wall_loc, 3] = wall_loc

# Task
num_mice = 1
num_sessions = 1
num_trials = 1
trials_before_wall = 0

# Build graph object
s0_dist = np.zeros(num_states)
s0_dist[0] = 1
sagiv_maze = DiGraph(num_states, edges, init_state_dist=s0_dist)
T = sagiv_maze.transitions
all_experiences = sagiv_maze.get_all_transitions()

sagiv_maze_wall = DiGraph(num_states, wall_edges, init_state_dist=s0_dist)
T_wall = sagiv_maze_wall.transitions
all_experiences_wall = sagiv_maze_wall.get_all_transitions()

# Build the goal dynamics matrix
gdm = build_gdm(0.05)

# Define MDP-related parameters
goal_states = goal_ids

## Build agent
episode_disc_rate = 0.90
num_replay_steps = 30

decay_rate = 0.99
noise = 0.00  # Update noise

alpha = 1.00
policy_temperature = 0.01
use_softmax = True
replay_mode = 'full'

## Run task
print(bn_ids)

# Build storage variables
# choices = np.zeros((num_mice, num_sessions, num_trials))
rewards = np.zeros((num_mice, num_sessions, num_trials))
goal_seq = np.zeros((num_mice, num_sessions, num_trials))

postoutcome_replays = np.zeros((num_mice, num_sessions, num_trials, num_replay_steps, 3)) - 1   # so obvious if some row is unfilled
prechoice_replays = np.zeros((num_mice, num_sessions, num_trials, num_replay_steps, 3)) - 1   # so obvious if some row is unfilled
hit_the_wall_replays = np.zeros((num_mice, num_sessions, num_replay_steps, 3)) - 1   # so obvious if some row is unfilled
states = np.empty((num_mice, num_sessions, num_trials), dtype=object)

init_goal_dist = np.zeros(num_goals)
init_goal_dist[0] = 1

start = time.time()
for mouse in range(num_mice):
    ga = SftmxGeodesicAgent(num_states, num_actions, goal_states, alpha=alpha, goal_dist=None, s0_dist=s0_dist, T=T,
                            policy_temperature=policy_temperature)

    ga.remember(all_experiences, overwrite=True)
    curr_maze = sagiv_maze
    true_GR = curr_maze.solve_GR(num_iters=500, gamma=ga.gamma)
    true_GR[:, :, [i for i in range(num_states) if i not in goal_states]] = 0  # Wipe out GR values for non-goal states
    ga.initialize_GR(true_GR)

    for session in range(num_sessions):
        goal_dist = init_goal_dist.copy()
        has_done_wall_replay = False
        for trial in range(num_trials):
            print(mouse, session, trial)

            # Pick a new goal for the trial
            if trial != trials_before_wall:
                goal_idx = np.random.choice(num_goals, p=goal_dist)
            else:
                curr_maze = sagiv_maze_wall
                goal_idx = 0

            goal_state = goal_ids[goal_idx]
            goal_seq[mouse, session, trial] = goal_state

            rvec = np.zeros(num_states)
            rvec[goal_state] = 1

            goal_dist = np.zeros(num_goals)
            goal_dist[goal_idx] = 1

            # Reset the agent location
            ga.curr_state = 0
            trial_states = [0]

            # Behave
            nsteps = 0
            while ga.curr_state not in goal_states:
                # Do stuff
                if trial == trials_before_wall and ga.curr_state == 0:
                    action = np.argmax(ga.policies[goal_state][ga.curr_state])
                    next_state, reward = curr_maze.step(ga.curr_state, action=action, reward_vector=rvec)
                else:
                    action, next_state, reward = curr_maze.step(ga.curr_state, policy=ga.policies[goal_state],
                                                                reward_vector=rvec)

                ga.basic_learn((ga.curr_state, action, next_state), decay_rate=decay_rate)  # Update GR

                # When encountering the wall for the first time, wipe the GR and do replay
                if trial == trials_before_wall and ga.curr_state == wall_loc and action == 3 and \
                        not has_done_wall_replay:
                    print('replay started')
                    # Reset the agent with new transition matrix
                    ga = SftmxGeodesicAgent(num_states, num_actions, goal_states, alpha=alpha, goal_dist=None,
                                            s0_dist=s0_dist, T=T_wall,
                                            policy_temperature=policy_temperature)
                    ga.curr_state = wall_loc
                    ga.remember(all_experiences_wall, overwrite=True)

                    # Replay
                    replays, (state_needs, transition_needs, gains, all_MEVBs, all_DEVBs), _ = ga.dynamic_replay(num_replay_steps, goal_states, gdm, goal_dist, verbose=True,
                                                      check_convergence=False,
                                                      prospective=False, disc_rate=episode_disc_rate)

                    hit_the_wall_replays[mouse, session, :, :] = replays
                    has_done_wall_replay = True

                ga.curr_state = next_state
                nsteps += 1
                trial_states.append(ga.curr_state)

            # Store trial reward
            rewards[mouse, session, trial] = reward
            states[mouse, session, trial] = trial_states

            # Post-outcome replay
            replays, _, _ = ga.dynamic_replay(num_replay_steps, goal_states, gdm, goal_dist, verbose=True,
                                              check_convergence=False,
                                              prospective=False, disc_rate=episode_disc_rate)

            postoutcome_replays[mouse, session, trial, :, :] = replays

            # Evolve goal distribution
            goal_dist = gdm @ goal_dist


# Save everything
np.savez('./Data/prediction/prediction.npz', posto=postoutcome_replays, prec=prechoice_replays,
         hit_the_wall=hit_the_wall_replays, goal_seq=goal_seq, wall_loc=wall_loc,
         bn_ids=bn_ids, bn_arm_start_ids=bn_arm_start_ids, states=states, goal_ids=goal_ids)
