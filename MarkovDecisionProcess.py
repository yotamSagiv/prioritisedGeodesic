from RL_utils import compute_occupancy

import numpy as np
np.set_printoptions(precision=3, suppress=True)


class MarkovDecisionProcess(object):
	"""
		The MarkovDecisionProcess class allows agents to interact with an environment.
		Users should provide the environment properties:
			* State-action transition matrix `transitions`
			* Action set [0, 1, ..., num_actions - 1]
			* Initial state distribution `init_state_distribution

		A reward vector is not necessary, but may be provided at each of the interaction 
		functions in order to add a reward component. Rewards will be assigned on the
		basis of arrival to a state.
	"""

	def __init__(self, transitions, num_actions, init_state_distribution=None):
		"""
		Initialize the MarkovDecisionProcess object with the relevant environment parameters.

		Args:
			transitions (np.ndarray): Transition matrix for the underlying MDP, T[s, a, g] = P(g | s, a)
			num_actions (int): Number of actions available at each state for the MDP.
			init_state_distribution (np.ndarray): Distribution over initial states for any given MDP episode.
		"""

		self.num_states = transitions.shape[0]
		self.num_actions = num_actions
		self.transitions = transitions

		if init_state_distribution is None:
			self.s0_dist = np.ones(self.num_states) / self.num_states
		else:
			self.s0_dist = init_state_distribution

	def sample_trajectories(self, num_trajectories, length, policy, reward_vector=None):
		"""
		Sample multiple trajectories of a given length by simulating a policy on the MDP.

		Args:
			num_trajectories (int): Number of trajectories to simulate.
			length (int): Length of each trajectory.
			policy (np.ndarray): Behavioural policy for the agent. Shape: (num_states, num_actions)
			reward_vector (np.ndarray): Vector describing the reward associated with each state.

		Returns:
			state_seqs (np.ndarray): Sequences of states visited across all trajectories.
			action_seqs (np.ndarray): Sequences of actions taken across all trajectories.
			reward_seqs (np.ndarray): Sequences of rewards received across all trajectories.
		"""
		state_seqs = np.zeros((num_trajectories, length), dtype=int)
		action_seqs = np.zeros((num_trajectories, length), dtype=int)  # last action is meaningless
		reward_seqs = np.zeros((num_trajectories, length))

		for i in range(num_trajectories):
			state_seq, action_seq, reward_seq = self.sample_trajectory(length, policy, reward_vector)
			state_seqs[i, :] = state_seq
			action_seqs[i, :] = action_seq
			reward_seqs[i, :] = reward_seq

		return state_seqs, action_seqs, reward_seqs

	def sample_trajectory(self, length, policy, reward_vector=None):
		"""
		Sample one trajectory of a given length by simulating a policy on the MDP.

		Args:
			length (int): Length of the trajectory.
			policy (np.ndarray): Behavioural policy for the agent. Shape: (num_states, num_actions)
			reward_vector (np.ndarray): Vector describing the reward associated with each state.

		Returns:
			state_seq (np.ndarray): Sequence of states visited.
			action_seq (np.ndarray): Sequence of actions taken.
			reward_seq (np.ndarray): Sequence of rewards received.
		"""
		state_seq = np.zeros(length, dtype=int)
		action_seq = np.zeros(length, dtype=int)
		reward_seq = np.zeros(length)

		# Sample initial state
		state_seq[0] = self.sample_initial_state()

		# Sample subsequent states
		for t in range(1, length + 1):
			action, next_state, reward = self.execute_policy(state_seq[t - 1], policy, reward_vector)

			if t < length:
				state_seq[t] = next_state

			action_seq[t - 1] = action
			reward_seq[t - 1] = reward

		return state_seq, action_seq, reward_seq

	def sample_initial_state(self):
		"""
		Sample the initial state distribution to pick an initial state.
		"""
		return np.random.choice(self.num_states, p=self.s0_dist)

	def step(self, state, policy=None, action=None, reward_vector=None):
		"""
		Execute a single step of the MDP simulation process. Either a specific action must be provided,
		or a policy from which an action may be selected.

		Args:
			state (int): The agent's current state.
			policy (np.ndarray): The agent's behavioural policy. Shape: (num_states, num_actions)
			action (int): The action the agent has taken.
			reward_vector (np.ndarray): Vector describing the reward associated with each state.

		Returns:
			(action), next_state, reward: The results of executing this step of the MDP. The action is optionally
				returned only when the user has provided a policy instead of an action directly.
		"""
		assert(policy is not None or action is not None), 'Buddy, you gotta give me at least an action or a policy.'

		if reward_vector is None:
			reward_vector = np.zeros(self.num_states)

		if policy is not None:
			return self.execute_policy(state, policy, reward_vector)

		if action is not None:
			return self.perform_action(state, action, reward_vector)

	def perform_action(self, state, action, reward_vector=None):
		"""
		Perform an action in a state, observe the resultant state and reward.

		Args:
			state (int): The agent's current state.
			action (int): The action undertaken by the agent.
			reward_vector (np.ndarray): Vector describing the reward associated with each state.

		Returns:
			next_state (int): The successor state observed after taking the action.
			reward (float): The received reward.
		"""
		next_state = np.random.choice(self.num_states, p=self.transitions[state, action, :])
		if reward_vector is not None:
			reward = reward_vector[next_state]
		else:
			reward = 0

		return next_state, reward

	def execute_policy(self, state, policy, reward_vector=None):
		"""
		Execute a policy in a given state.

		Args:
			state (int): The agent's current state.
			policy (np.ndarray): The agent's behavioural policy. Shape: (num_states, num_actions)
			reward_vector (np.ndarray): Vector describing the reward associated with each state.

		Returns:
			action (int): The action sampled from the policy.
			next_state (int): The successor state observed after taking the action.
			reward (float): The received reward.
		"""
		action = np.random.choice(self.num_actions, p=policy[state])
		next_state, reward = self.perform_action(state, action, reward_vector)
		return action, next_state, reward

	def solve_GR(self, num_iters, gamma, conv_tol=1e-6, filter_self_transitions=False) -> np.ndarray:
		"""
		Solve the MDP and return the true GR for it. This is done using a Geodesic analogue
		of the value iteration algorithm.

		Args:
			num_iters (int): Maximum number of iterations for the value iteration algorithm.
			gamma (float): Temporal discount factor.
			conv_tol(float): Early stopping criterion. If no state changes by more than conv_tol, the
				GR is assumed to have converged and the algorithm is stopped.

		Returns:
			update_G (np.ndarray): The true GR for this MDP.
		"""
		Gs = [np.zeros((self.num_states, self.num_actions, self.num_states)),
			  np.zeros((self.num_states, self.num_actions, self.num_states))]
		update_G = None

		for i in range(num_iters):
			ref_G = Gs[i % 2]
			update_G = Gs[1 - (i % 2)]

			for s in range(self.num_states):
				for goal in range(self.num_states):
					for a in range(self.num_actions):
						dG = 0
						for sp in range(self.num_states):
							if filter_self_transitions and sp == s:
								continue

							if sp == goal:
								dG += self.transitions[s, a, sp]
							else:
								dG += self.transitions[s, a, sp] * gamma * np.max(ref_G[sp, :, goal])

						update_G[s, a, goal] = dG

			# Check for early convergence
			if np.all(np.abs(update_G - ref_G) <= conv_tol):
				break

		return update_G

	def is_connected(self):
		uniform_policy = np.ones((self.num_states, self.num_actions))
		M = compute_occupancy(uniform_policy, self.transitions)

		return 0 not in M

	def get_all_transitions(self, tol=1e-6, filter_self_transitions=True, rvec=None):
		"""
			Return a list of all (s, a, s') tuples for which P(s' | s, a) >= tol.
		"""
		experiences = []
		for start in range(self.num_states):
			for action in range(self.num_actions):
				for successor in range(self.num_states):
					if filter_self_transitions and start == successor:  # Disallows replay of invalid transitions
						continue

					if self.transitions[start, action, successor] >= tol and rvec is not None:
						experiences.append((start, action, successor, rvec[successor]))
					elif self.transitions[start, action, successor] >= tol:
						experiences.append((start, action, successor))

		return experiences

	def update_transitions(self, new_transitions):
		"""
			Update the current transition matrix to a new one.

		Args:
			new_transitions (np.ndarray): THe new transition matrix
		"""
		self.transitions = new_transitions



####### Testing script
if __name__ == '__main__':
	nstates = 4   # 0: top left, 1 : top right, 2: bottom left, 3: bottom right
	nactions = 4  # 0: left, 1: up, 2: right, 3: down
	
	# Set up transition matrix as deterministic
	T = np.zeros((nstates, nactions, nstates))
	T[0, 0, :] = [1, 0, 0, 0]
	T[0, 1, :] = T[0, 0, :]
	T[0, 2, :] = [0, 1, 0, 0]
	T[0, 3, :] = [0, 0, 1, 0]

	T[1, 0, :] = [1, 0, 0, 0]
	T[1, 1, :] = [0, 1, 0, 0]
	T[1, 2, :] = T[1, 1, :]
	T[1, 3, :] = [0, 0, 0, 1]

	T[2, 0, :] = [0, 0, 1, 0]
	T[2, 1, :] = [1, 0, 0, 0]
	T[2, 2, :] = [0, 0, 0, 1]
	T[2, 3, :] = T[2, 0, :]

	T[3, 0, :] = [0, 0, 1, 0]
	T[3, 1, :] = [0, 1, 0, 0]
	T[3, 2, :] = [0, 0, 0, 1]
	T[3, 3, :] = [0, 0, 0, 1]

	# Does the init work?
	mdp = MarkovDecisionProcess(T, nactions)

	# Can we sample trajectories?
	pi = (1 / nactions) * np.ones((nstates, nactions))
	s_seqs, a_seqs, r_seqs = mdp.sample_trajectories(10, 15, pi, reward_vector=np.array([1, 0, 1, 1]))

	print(s_seqs)
	print(a_seqs)
	print(r_seqs)
