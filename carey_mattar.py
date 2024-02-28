import numpy as np

from graph import DiGraph
from reward_agent import RewardAgent, SftmxRewardAgent
from RL_utils import softmax

np.random.seed(865612)

### Convenience functions
def action_seq(goal, arm_length, start_to_choice):
    """
    Given maze parameters, given the full sequence of actions that leads from the start state to the goal state.

    Args:
        goal ():
        arm_length ():
        start_to_choice ():

    Returns:

    """
    actions = []

    # Get to choice point
    for _ in range(start_to_choice - 1):
        actions.append(1)

    # Get to goal arm
    actions.append(goal + 2)

    # Continue to the end
    for _ in range(arm_length - 1):
        actions.append(1)

    return actions

### Main script


## Set parameters
# Geometry
num_arms = 2
arm_length = 2
start_to_choice = 2

## Build underlying MDP
# Basics
num_states = num_arms * arm_length + start_to_choice
num_actions = 2 + num_arms
permit_backtrack = True  # If the agent makes a choice, can it go backwards to the choice point?

# Fill out edges
edges = np.zeros((num_states, num_actions))
for state in range(num_states):  # Convenient way of filling in null actions
    edges[state, :] = state

# Overwrite null actions with actual transitions
for state in range(start_to_choice):
    if not state == 0:
        edges[state, 0] = state - 1  # Go left

    if not state == start_to_choice - 1:
        edges[state, 1] = state + 1  # Go right

# Add choice transitions
for action in range(2, num_actions):
    edges[start_to_choice - 1, action] = start_to_choice + (action - 2) * arm_length

# Add within-arm transitions
for arm in range(num_arms):
    for rel_state in range(arm_length):
        state = start_to_choice + arm * arm_length + rel_state

        if not rel_state == 0:
            edges[state, 0] = state - 1  # Backwards through arm
        elif permit_backtrack:
            edges[state, 0] = start_to_choice - 1

        if not rel_state == arm_length - 1:
            edges[state, 1] = state + 1  # Forwards through arm


# Task
num_mice = 10
num_sessions = 10
num_trials = 20

# Build graph object
s0_dist = np.zeros(num_states)
s0_dist[0] = 1
carey_maze = DiGraph(num_states, edges, init_state_dist=s0_dist)
T = carey_maze.transitions
all_experiences = carey_maze.get_all_transitions()

# Define MDP-related parameters
goal_states = start_to_choice + np.arange(num_arms) * arm_length + (arm_length - 1)

## Build agent
behav_lr = 1.0
behav_temp = 0.5

num_replay_steps = 3
replay_lr = 0.7
replay_temp = 0.15

decay_rate = 0.75
noise = 0.00  # Update noise

behav_alpha = 1.0
replay_alpha = 0.7
policy_temperature = 0.1
use_softmax = True
replay_mode = 'full'

## Run task

# Build storage variables
choices = np.zeros((num_mice, num_sessions, num_trials))
rewards = np.zeros((num_mice, num_sessions, num_trials))
sess_seq = np.zeros((num_mice, num_sessions, num_trials))

postoutcome_replays = np.zeros((num_mice, num_sessions, num_trials, num_replay_steps, 4)) - 1   # so obvious if some row is unfilled
states_visited = np.zeros((num_mice, num_sessions, num_trials, start_to_choice + arm_length))

for mouse in range(num_mice):
    # Agent parameters
    behav_goal_vals = np.zeros(num_arms)
    replay_goal_vals = np.zeros(num_arms)

    # Instantiate agent
    if not use_softmax:
        ga = RewardAgent(num_states, num_actions, alpha=behav_alpha, s0_dist=s0_dist, T=T)
    else:
        ga = SftmxRewardAgent(num_states, num_actions, alpha=behav_alpha, s0_dist=s0_dist, T=T,
                                policy_temperature=policy_temperature)

    # Simulate
    sess_type = 0
    for session in range(num_sessions):
        # Set basic task variables
        rvec = np.zeros(num_states)
        if sess_type == 0:
            rvec[goal_states[0]] = 1.5
            rvec[goal_states[1]] = 1

            behav_goal_vals[0] = 1.5
            behav_goal_vals[1] = 1
        else:
            rvec[goal_states[1]] = 1.5
            rvec[goal_states[0]] = 1

            behav_goal_vals[1] = 1.5
            behav_goal_vals[0] = 1

        exps_with_rewards = [(s, a, sp, rvec[sp]) for (s, a, sp) in all_experiences]
        ga.remember(exps_with_rewards, overwrite=True)

        for trial in range(num_trials):
            if trial % 50 == 0:
                print('mouse %d, session %d, trial %d' % (mouse, session, trial))

            ga.curr_state = 0
            sess_seq[mouse, session, trial] = sess_type

            # Choose an arm
            chosen_arm = np.random.choice(num_arms, p=softmax(behav_goal_vals, behav_temp))
            choices[mouse, session, trial] = chosen_arm

            # Go to it
            act_seq = action_seq(chosen_arm, arm_length, start_to_choice)
            for adx, action in enumerate(act_seq):
                next_state, reward = carey_maze.step(ga.curr_state, action=action, reward_vector=rvec)
                ga.basic_learn((ga.curr_state, action, next_state, reward), decay_rate=decay_rate, noise=noise)  # Update GR

                ga.curr_state = next_state
                states_visited[mouse, session, trial, adx + 1] = ga.curr_state

                # Check for rewards
                if adx == len(act_seq) - 1:
                    behav_goal_vals[chosen_arm] += behav_lr * (reward - behav_goal_vals[chosen_arm])
                    replay_goal_vals[chosen_arm] += replay_lr * (reward - replay_goal_vals[chosen_arm])
                    rewards[mouse, session, trial] = reward

            # Post-outcome replay
            replays, _, _ = ga.replay(num_replay_steps, verbose=True, prospective=True, alpha=replay_alpha)
            postoutcome_replays[mouse, session, trial, :, :] = replays

        sess_type = int(1 - sess_type)

# Save everything
np.savez('./Data/carey/mattar/Q.npz', posto=postoutcome_replays, state_trajs=states_visited,
         choices=choices, rewards=rewards, sess_seq=sess_seq, allow_pickle=True)
