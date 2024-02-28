import numpy as np
import matplotlib.pyplot as plt
import sys

from graph import DiGraph
from geodesic_agent import GeodesicAgent, SftmxGeodesicAgent
from RL_utils import softmax

np.random.seed(865612)


#### Convenience functions
def action_seq(goal, arm_length, start_to_choice):
	"""
		For a given goal, return the sequence of actions that optimally bring the agent from
		the start state to that goal, given values for how long arms are (`arm_length`) and the distance
		between the start state and the choice point (`start_to_choice`).
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


## Set parameters
# Geometry
num_arms = 8
arm_length = 2
start_to_choice = 2

## Build underlying MDP
# Basics
num_states = num_arms * arm_length + start_to_choice
num_actions = 2 + num_arms

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

		if not rel_state == arm_length - 1: 
			edges[state, 1] = state + 1  # Forwards through arm

# Task
num_sessions = 10
num_sessions = 200
num_trials = 200
reward_thresh = 15

# Build graph object
s0_dist = np.zeros(num_states)
s0_dist[0] = 1
gillespie_maze = DiGraph(num_states, edges, init_state_dist=s0_dist)
T = gillespie_maze.transitions
all_experiences = gillespie_maze.get_all_transitions()

# Define MDP-related parameters
goal_states = start_to_choice + np.arange(num_arms) * arm_length + (arm_length - 1)

## Build agent
behav_lr = 1  # Behavioural module learning rate
behav_temp = 0.3

num_replay_steps = 3
replay_lr = 0.30  # Goal distribution value learning rate
replay_temp = 0.20  # Goal distribution softmax inverse temperature

replay_decay = 0.90  # Decay of the replay Q-value vector
decay_rate = 0.95  # GR decay rate on every step
noise = 0.00  # Update noise

behav_alpha = 1.0  # Online GR learning rate
replay_alpha = 0.7  # Replay GR learning rate
policy_temperature = 0.1
use_softmax = True
replay_mode = 'full'

## Run task

# Build storage variables
choices = np.zeros((num_sessions, num_trials))
rewards = np.zeros((num_sessions, num_trials))
goal_seq = np.zeros((num_sessions, num_trials))

prechoice_replays = np.zeros((num_sessions, num_trials, num_replay_steps, 3)) - 1     # No states or actions are equal to -1
postoutcome_replays = np.zeros((num_sessions, num_trials, num_replay_steps, 3)) - 1   # so obvious if some row is unfilled
states_visited = np.zeros((num_sessions, num_trials, start_to_choice + arm_length))

# Simulate
for session in range(num_sessions):
	# Reset basic task variables
	reward_counter = 0
	active_goal = np.random.choice(goal_states)
	rvec = np.zeros(num_states)
	rvec[active_goal] = 1

	# Reset agent parameters
	behav_goal_vals = np.zeros(num_arms)
	replay_goal_vals = np.zeros(num_arms)

	# Reset agent
	if not use_softmax:
		ga = GeodesicAgent(num_states, num_actions, goal_states, alpha=behav_alpha, goal_dist=None, s0_dist=s0_dist, T=T)
	else:
		ga = SftmxGeodesicAgent(num_states, num_actions, goal_states, alpha=behav_alpha, goal_dist=None, s0_dist=s0_dist, T=T,
								policy_temperature=policy_temperature)

	ga.remember(all_experiences)
	true_GR = gillespie_maze.solve_GR(num_iters=500, gamma=ga.gamma)
	ga.initialize_GR(true_GR)

	for trial in range(num_trials):
		if trial % 50 == 0:
			print('session %d, trial %d' % (session, trial))

		ga.curr_state = 0
		goal_seq[session, trial] = active_goal

		# Choose an arm
		chosen_arm = np.random.choice(num_arms, p=softmax(behav_goal_vals, behav_temp))
		choices[session, trial] = chosen_arm

		# Go to it
		act_seq = action_seq(chosen_arm, arm_length, start_to_choice)
		for adx, action in enumerate(act_seq):
			next_state, reward = gillespie_maze.step(ga.curr_state, action=action, reward_vector=rvec)
			ga.basic_learn((ga.curr_state, action, next_state), decay_rate=decay_rate, noise=noise)  # Update GR

			ga.curr_state = next_state
			states_visited[session, trial, adx + 1] = ga.curr_state

			# Check for rewards
			if adx == len(act_seq) - 1:
				# Update the value functions for each arm as Rescorla-Wagner + forgetting (linear interpolation form)
				replay_goal_vals[np.arange(len(replay_goal_vals)) != chosen_arm] *= replay_decay
				replay_goal_vals[chosen_arm] += replay_lr * (reward - replay_goal_vals[chosen_arm])

				behav_goal_vals[chosen_arm] += behav_lr * (reward - behav_goal_vals[chosen_arm])

				reward_counter += reward
				rewards[session, trial] = reward

		# Post-outcome replay
		if reward_counter == 1:
			pass
		elif reward_counter == reward_thresh - 1:
			pass

		replays, _, _ = ga.replay(num_replay_steps, goal_dist=softmax(replay_goal_vals, replay_temp), verbose=True,
								  check_convergence=False, prospective=True, EVB_mode=replay_mode, alpha=replay_alpha)
		postoutcome_replays[session, trial, :, :] = replays

		# Update goal?
		if reward_counter >= reward_thresh:
			new_goals = np.delete(goal_states, np.argwhere(goal_states == active_goal))
			active_goal = np.random.choice(new_goals)

			rvec = np.zeros(num_states)
			rvec[active_goal] = 1
			reward_counter = 0

# Save everything
np.savez('./Data/gillespie/GR/GR.npz', prec=prechoice_replays, posto=postoutcome_replays, state_trajs=states_visited,
		 choices=choices, rewards=rewards, goal_seq=goal_seq, num_sessions=num_sessions, num_trials=num_trials,
		 trial=trial)

