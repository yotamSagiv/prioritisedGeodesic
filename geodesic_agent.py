import numpy as np
from RL_utils import dynamics_policy_onestep, softmax

## TODO: Implement prospective need evaluation by sampling
## TODO: think about the need term in dynamic replay (does it take into account being at the curr goal for next goal?)
class GeodesicAgent(object):
	"""
		The GeodesicAgent class solves MDPs using the Geodesic Representation (GR),
		which is a control version of the Successor Representation (SR). Unlike the SR,
		which develops a matrix of future state occupancy under a given policy, the GR is an off-policy
		estimate of distances between states. In deterministic regimes, for example, it evaluates to

			G[i, a, j] = gamma^d(i, a, j)

		where gamma is a discounting factor and d(i, a, j) is the length of the shortest path from `i` to `j`
		after taking action `a`.
	"""

	def __init__(self, num_states: int, num_actions: int, goal_states: np.ndarray,
				 T: np.ndarray,
				 goal_dist: np.ndarray = None,
				 s0_dist: np.ndarray = None,
				 alpha: float = 0.3,
				 gamma: float = 0.95,
				 min_gain: float = 0):
		"""
		Construct the GeodesicAgent object.

		Args:
			num_states (int): Number of states in the underlying MDP
			num_actions (int): Number of actions available at each state
			goal_states (np.ndarray): A subset of states that are marked as "goals"
			T (np.ndarray): The one-step transition matrix for the MDP.
				T[s, a, g] gives the probability of transitioning from state s to state g after
				taking action a.
			goal_dist (np.ndarray): The distribution over which goals are most likely to manifest
			s0_dist (np.ndarray): Initial state distribution
			alpha (float): Learning rate
			gamma (float): Temporal discount rate
			min_gain (float): Minimum value for gain computation
		"""

		# MDP properties
		self.num_states = num_states
		self.num_actions = num_actions
		self.goal_states = goal_states
		self.curr_state = -1  # Pre-initial state indicating action has not yet started
		if goal_dist is None:
			goal_dist = np.ones(len(goal_states)) / len(goal_states)
		self.goal_dist = goal_dist
		self.T = T
		self.s0_dist = s0_dist

		# Agent properties
		self.alpha = alpha  # Learning rate
		self.gamma = gamma
		self.min_gain = min_gain
		self.G = np.zeros(
			(num_states, num_actions, num_states))  # Geodesic representation matrix is (roughly) gamma^shortest_path
		self.memory = []  # Memory bank for later replay

		# Separate policies for each goal state, each initialised as uniform
		uniform_policy = np.ones((num_states, num_actions)) / num_actions
		self.policies = {goal_states[i]: uniform_policy.copy() for i in range(len(goal_states))}

		# Separate transition structures for each of the modified MDPs, one per goal
		self.mod_Ts = {}
		for i in range(len(goal_states)):
			goal = goal_states[i]
			mod_T = self.modify_transition_mat(T, goal)

			self.mod_Ts[goal] = mod_T

	def modify_transition_mat(self, T, goal):
		mod_T = np.zeros((self.num_states + 1, self.num_actions, self.num_states + 1))

		# Everything about the modified transition matrix is the same except it has an extra junk state
		# with a Prob = 1 self-loop, and the goal has a transition to it with Prob = 1
		mod_T[:self.num_states, :, :self.num_states] = T  # same
		mod_T[-1, :, -1] = 1  # self-loop
		mod_T[goal, :, :] = 0
		mod_T[goal, :, -1] = 1  # goal->junk state transition

		return mod_T

	def initialize_GR(self, new_G, update_policies=True):
		"""
		Replace the agent's current Geodesic representation with a new one. Optionally,
		update all of its policies to be consistent with this new GR.

		Args:
			new_G (np.ndarray): The new Geodesic representation.
			update_policies (boolean): If True, the agent will also update its policies to be consistent
				with the new GR.

		Returns:
			None
		"""
		self.G = new_G.copy()
		if update_policies:
			for goal in self.goal_states:
				self.derive_policy(goal, set_policy=True)

	def derive_policy(self, goal_state, G=None, set_policy=False, epsilon=0):
		"""
		Derive the policy for reaching a given goal state. Since
		the GR represents the shortest (expected) paths, we can
		simply take the max at every state.

		Allow an epsilon-greedy addition to facilitate exploration.

		Args:
			goal_state (int): The goal with respect to which the policy is derived.
			G (np.ndarray, optional): The Geodesic representation over which the policy is derived.
				If none is provided, the agent's current one will be used.
			set_policy (boolean): If True, the computed policy will update the agent's current policy
				 for the specified goal.
			epsilon (float): Epsilon parameter for epsilon-greedy action policy.

		Returns:
			policy (np.ndarray): The computed policy.
		"""

		# Allow arbitrary G to be fed in, default to current G
		if G is None:
			G = self.G

		policy = np.zeros((self.num_states, self.num_actions))

		# Compute policy
		for state in range(self.num_states):
			best_actions = np.flatnonzero(G[state, :, goal_state] == np.max(G[state, :, goal_state]))
			num_best_actions = len(best_actions)  # Split 1 - epsilon ties equally

			policy[state, :] = epsilon / self.num_actions
			policy[state, best_actions] += (1 - epsilon) / num_best_actions  # Deterministic for epsilon = 0

		# Cache if wanted
		if set_policy:
			self.policies[goal_state] = policy

		return policy

	def update_state_policy(self, state, goal_state, G=None, set_policy=False, epsilon=0):
		"""
		Update the internal policy only for a given state.

		Allow an epsilon-greedy addition to facilitate exploration.

		Args:
			state (int): The state receiving the policy update
			goal_state (int): The state with respect to which the policy is being updated
			G (np.ndarray, optional): The Geodesic representation over which the policy is derived.
				If none is provided, the agent's current one will be used.
			set_policy (boolean): If True, the computed policy will update the agent's current policy
				 for the specified goal.
			epsilon (float): Epsilon parameter for epsilon-greedy action policy.

		Returns:
			policy (np.ndarray): The computed policy.
		"""

		# Allow arbitrary G to be fed in, default to current G
		if G is None:
			G = self.G

		# Compute policy
		best_actions = np.flatnonzero(G[state, :, goal_state] == np.max(G[state, :, goal_state]))
		num_best_actions = len(best_actions)  # Split 1 - epsilon ties equally

		if not set_policy:  # Only re-copy the whole thing if we're not planning on saving it.
			policy = self.policies[goal_state].copy()
		else:
			policy = self.policies[goal_state]

		policy[state, :] = epsilon / self.num_actions  # Deterministic for epsilon = 0
		policy[state, best_actions] += (1 - epsilon) / num_best_actions

		# Cache if wanted
		if set_policy:
			self.policies[goal_state] = policy

		return policy

	def remember(self, transitions, overwrite=False):
		"""
			Add a set of transitions to the memory bank.

			Args:
				transitions (list): The list of memories to be added to the memory bank.
					Each memory must be a tuple (s, a, g), indicating that action a was
					taken in state s and reached state g.
		"""
		if overwrite:
			self.memory = []

		for transition in transitions:
			if transition not in self.memory:
				self.memory.extend([transition])

	def forget(self, transitions, verbose=False):
		"""
			Remove a set of transitions from the memory bank.

			Args:
				transitions (list): The list of memories to be removed from the memory bank.
					Each memory must be a tuple (s, a, g), indicating that action a was
					taken in state s and reached state g.
				verbose (boolean): If set to True, forget() will print a message indicating if a transition
					was not found in memory.
		"""
		for transition in transitions:
			if transition in self.memory:
				self.memory.remove(transition)
			elif verbose:
				print('transition', transition, ' not located in memory.')

	def decay(self, decay_rate, update_policies=True):
		self.G *= decay_rate

		# Consolidate policies according to updated GR
		if update_policies:
			for goal in self.goal_states:
				self.policies[goal] = self.derive_policy(goal)

	def basic_learn(self, transition, goal_states=None, decay_rate=None, noise=0, update_policies=True, alpha=None):
		"""
			One-step transition learning, with forgetting.

			Args:
				transition ():
				goal_states ():
				decay_rate ():
				noise ():
				update_policies():
		"""
		if decay_rate is None:
			decay_rate = 1  # No forgetting
		if alpha is None:
			alpha = self.alpha

		s_k, a_k, s_kp = transition
		dG = np.zeros_like(self.G)
		mG = np.ones_like(self.G) * decay_rate

		if goal_states is None:
			goal_states = self.goal_states

		# For each goal...
		for gdx, goal in enumerate(goal_states):
			# Compute GR delta wrt this goal
			# Will use linear interpolation form of the GR update equation, so no need for prediction error
			if s_kp == goal:
				GR_value = 1
			else:
				GR_value = self.gamma * np.max(self.G[s_kp, :, goal])

			# Consolidate deltas
			dG[s_k, a_k, goal] += GR_value
			mG[s_k, a_k, goal] *= (1 - alpha)

		# Update GR according to interpolation form of the update equation
		self.G = mG * self.G + alpha * dG

		# Potentially add noise
		if noise > 0:
			self.G += np.random.normal(0, noise, size=self.G.shape)

		# Consolidate policies according to updated GR
		if update_policies:
			for goal in self.goal_states:
				self.policies[goal] = self.derive_policy(goal)

	def compute_EVB_vector(self, goal_states, replay_seq, prospective=False, verbose=False, mode='full', alpha=None):
		"""
		Compute the EVB vector EVB(s, :, e_k) -- that is, the expected value of backing up `e_k` for state `s` with
		respect to all possible goals.

		Args:
			goal_states (np.ndarray): The list of goals.
			replay_seq (list): The previously executed replays.
			prospective (boolean): Controls whether the agent plans prospectively or using their current state.
				If prospective=False, the need term of EVB is computed with respect to the agent's current state.
				If prospective=True, the need term of EVB is computed with respect to the agent's initial
				state distribution.
			verbose (boolean): Controls whether various intermediate variables are returned at the end of the process.
			mode (string):

		Returns:
			MEVBs (np.ndarray): The vector of EVBs, with one entry per goal/transition pair.
			state_needs (np.ndarray): The need term, computed for every step of replay, under every possible goal,
				from all states, for all states.
			transition_needs (np.ndarray): The need term, computed for every step of replay, under every possible
				goal, for each transition.
			gains (np.ndarray): The gain term, computed for every step of replay, under every possible goal, for
				each transition.
		"""
		if alpha is None:
			alpha = self.alpha

		MEVBs = np.zeros((len(goal_states), len(self.memory)))  # Best transition is picked greedily at each step
		G_ps = {}  # At each replay step, cache G primes since they are goal-invariant
		dGs = {}   # Also cache the direct updates

		state_needs = None
		transition_needs = None
		gains = None
		if verbose:
			state_needs = np.zeros((len(goal_states), self.num_states, self.num_states))
			transition_needs = np.zeros((len(goal_states), len(self.memory)))
			gains = np.zeros((len(goal_states), len(self.memory)))

		# Compute EVB for all transitions across all goal states
		for gdx, goal in enumerate(goal_states):
			# If we have a policy cached, grab it. Otherwise, recompute fully.
			if goal in self.goal_states:
				policy = self.policies[goal]
			else:
				policy = self.derive_policy(goal)

			# Compute SR induced by this policy and the task dynamics
			M_pi = self.compute_occupancy(policy, self.mod_Ts[goal])[:self.num_states, :self.num_states]

			# Log, if wanted
			if verbose:
				state_needs[gdx, :, :] = M_pi

			# Compute EVB for each transition
			for tdx, transition in enumerate(self.memory):
				if tdx in G_ps.keys():
					G_p = G_ps[tdx]
					dG = dGs[tdx]
				else:
					dG, _ = self.compute_nstep_update(transition, replay_seq=replay_seq, goal_states=goal_states)
					dGs[tdx] = dG

					G_p = self.G + alpha * dG
					G_ps[tdx] = G_p

				if mode == 'full':
					need, gain, evb = self.compute_multistep_EVB(transition, goal, policy,
																 replay_seq,
																 curr_state=self.curr_state,
																 M=M_pi,
																 G_p=G_p,
																 prospective=prospective,
																 alpha=alpha)
				elif mode == 'sweeping':
					need, gain, evb = self.compute_sweeping_EVB(transition, goal,
																replay_seq,
																curr_state=self.curr_state,
																M=M_pi,
																dG=dG,
																prospective=prospective)


				MEVBs[gdx, tdx] = evb

				# Log quantities, if desired
				if verbose:
					gains[gdx, tdx] = gain
					transition_needs[gdx, tdx] = need

		if verbose:
			return MEVBs, (state_needs, transition_needs, gains)
		else:
			return MEVBs

	def uniform_replay(self, num_steps, alpha=None):
		replay_seq = []
		for step in range(num_steps):
			# Pick a memory at random
			mem = np.random.choice(len(self.memory))
			replay_seq.append(self.memory[mem])

			# Learn
			self.nstep_learn(replay_seq, alpha=alpha)

	def replay(self, num_steps, goal_states=None, goal_dist=None, prospective=False, verbose=False,
			   check_convergence=True, convergence_thresh=0.0, otol=1e-6, learn_seq=None, EVB_mode='full', alpha=None):
		"""
		Perform replay, prioritised under a (meta-)expected value of backup rule.
		Do this by iterating over all available transitions in memory, and averaging
		the EVBs over the list of potential future goal states.

		Args:
			num_steps (int): Maximum number of steps of replay to be performed.
			goal_states (np.ndarray): The set of particular goal states with respect to which replay should occur.
			goal_dist (np.ndarray): The distribution weighting those goals.
			prospective (boolean): Controls whether the agent plans prospectively or using their current state.
				If prospective=False, the need term of EVB is computed with respect to the agent's current state.
				If prospective=True, the need term of EVB is computed with respect to the agent's initial
				state distribution.
			verbose (boolean): Controls whether various intermediate variables are returned at the end of the process.
			check_convergence (boolean): Controls whether replay can end early if the Geodesic representation has
				converged.
			convergence_thresh (float): Tolerance on absolute mean change in the GR for convergence.
			otol (float): Ties in EVB are broken randomly. Otol defines the threshold for a tie.
			learn_seq (list): If provided, learn_seq stipulates the sequence of state to be replayed. All the EVB
				metrics are still computed for analysis purposes, but the outcome is ignored.
			EVB_mode (string):

		Returns:
			replay_seq (np.ndarray): The sequence of chosen replays.
			state_needs (np.ndarray): The need term, computed for every step of replay, under every possible goal,
				from all states, for all states.
			transition_needs (np.ndarray): The need term, computed for every step of replay, under every possible
				goal, for each transition.
			gains (np.ndarray): The gain term, computed for every step of replay, under every possible goal, for
				each transition.
			all_MEVBs (np.ndarray): The aggregate EVB, computed for every step of replay, under every possible goal,
				for each transition.
			backups (list): The full list of backed-up states, on every step of replay. Includes auxiliarry states
				updated through multistep backups.
		"""
		# Input validation, blah blah
		if goal_states is None:
			goal_states = self.goal_states
		if goal_dist is None:
			goal_dist = self.goal_dist

		# If verbose usage, build storage structures
		state_needs = None
		transition_needs = None
		gains = None
		all_MEVBs = None
		if verbose:
			state_needs = np.zeros((num_steps, len(goal_states), self.num_states, self.num_states))
			transition_needs = np.zeros((num_steps, len(goal_states), len(self.memory)))
			gains = np.zeros((num_steps, len(goal_states), len(self.memory)))
			all_MEVBs = np.zeros((num_steps, len(goal_states), len(self.memory)))

		# Start replaying
		replay_seq = []  # Maintain a list of replayed memories for use in multistep backups
		backups = []  # Maintain a list of transitions replayed in each backup step
		for step in range(num_steps):
			out = self.compute_EVB_vector(goal_states, replay_seq, prospective, verbose, EVB_mode, alpha=alpha)
			if verbose:
				MEVBs, (state_need, transition_need, gain) = out
				state_needs[step, :, :, :] = state_need
				transition_needs[step, :, :] = transition_need
				gains[step, :, :] = gain
				all_MEVBs[step, :, :] = MEVBs
			else:
				MEVBs = out

			# Average MEVBs, weighted by goal distribution
			MEVBs = np.average(MEVBs, axis=0, weights=goal_dist)

			# Pick the best one
			if learn_seq:
				best_memory = learn_seq[step]
			else:
				best_memories = np.argwhere(np.abs(MEVBs - np.max(MEVBs)) <= otol).flatten()
				best_memory = self.memory[np.random.choice(best_memories)]

			replay_seq.append(best_memory)

			# Learn!
			if check_convergence:
				backup, mag_delta = self.nstep_learn(replay_seq, ret_update_mag=True, alpha=alpha)
				if mag_delta <= convergence_thresh:  # Reached convergence
					# Cap the storage data structures, if necessary
					if verbose:
						state_needs = state_needs[:step, :, :, :]
						gains = gains[:step, :, :]
						all_MEVBs = all_MEVBs[:step, :, :]
						transition_needs = transition_needs[:step, :, :]

					break
			else:
				backup = self.nstep_learn(replay_seq, alpha=alpha)

			backups.append(backup)

		if verbose:
			return np.array(replay_seq), (state_needs, transition_needs, gains, all_MEVBs), backups

	def dynamic_replay(self, num_steps, goal_states, goal_dynamics, init_goal_dist, prospective=False,
					   verbose=False, check_convergence=True, convergence_thresh=0.0, otol=1e-6, learn_seq=None,
					   disc_rate=None, EVB_mode='full', alpha=None):
		"""
		Perform replay prioritized under a regime where the goal evolves through some dynamics process, described
		by a goal transition matrix. Like in replay(), this prioritization is accomplished by computing the
		expected utility of replaying all memories and picking the best one.

		The "dynamic" EVB of a given transition can be straightforwardly shown to be equal to:

			DEVB(s, e_k) = (I - gamma * T_g)^-1 g_0 \dot EVB(s, :, e_k)

		where `e_k` is a particular memory, `I` is the identity matrix, `gamma` is a temporal discounting factor,
		`T_g` is the transition matrix describing the goal evolution process, `g_0` is the initial goal distribution,
		and `EVB(s, :, e_k)` is the vector of EVBs for experience `e_k` with respect to all states.

		Args:
			num_steps (int): Maximum number of steps of replay to be performed.
			goal_states (np.ndarray): The list of possible goal states.
			goal_dynamics (np.ndarray): Matrix describing the goal evolution process.
			init_goal_dist (np.ndarray): The initial distribution over goal activation.
			prospective (boolean): Controls whether the agent plans prospectively or using their current state.
				If prospective=False, the need term of EVB is computed with respect to the agent's current state.
				If prospective=True, the need term of EVB is computed with respect to the agent's initial
				state distribution.
			verbose (boolean): Controls whether various intermediate variables are returned at the end of the process.
			check_convergence (boolean): Controls whether replay can end early if the Geodesic representation has
				converged.
			convergence_thresh (float): Tolerance on absolute mean change in the GR for convergence.
			otol (float): Ties in EVB are broken randomly. Otol defines the threshold for a tie.
			learn_seq (list): If provided, learn_seq stipulates the sequence of state to be replayed. All the EVB
				metrics are still computed for analysis purposes, but the outcome is ignored.
			disc_rate (float): Temporal discounting for whatever the time-step size is. Will be the agent's gamma
				parameter by default.

		Returns:
			replay_seq (np.ndarray): The sequence of chosen replays.
			state_needs (np.ndarray): The need term, computed for every step of replay, under every possible goal,
				from all states, for all states.
			transition_needs (np.ndarray): The need term, computed for every step of replay, under every possible
				goal, for each transition.
			gains (np.ndarray): The gain term, computed for every step of replay, under every possible goal, for
				each transition.
			all_MEVBs (np.ndarray): The aggregate EVB, computed for every step of replay, under every possible goal,
				for each transition.
			backups (list): The full list of backed-up states, on every step of replay. Includes auxiliary states
				updated through multistep backups.
		"""
		# Check for discount rate
		if disc_rate is None:
			disc_rate = self.gamma

		# If verbose usage, build storage structures
		state_needs = None
		transition_needs = None
		gains = None
		all_MEVBs = None
		all_DEVBs = None
		if verbose:
			state_needs = np.zeros((num_steps, len(goal_states), self.num_states, self.num_states))
			transition_needs = np.zeros((num_steps, len(goal_states), len(self.memory)))
			gains = np.zeros((num_steps, len(goal_states), len(self.memory)))
			all_MEVBs = np.zeros((num_steps, len(goal_states), len(self.memory)))
			all_DEVBs = np.zeros((num_steps, len(self.memory)))

		# Start replaying
		replay_seq = []  # Maintain a list of replayed memories for use in multistep backups
		backups = []  # Maintain a list of transitions replayed in each backup step
		weights = np.linalg.inv(np.eye(len(goal_states)) - disc_rate * goal_dynamics) @ init_goal_dist
		for step in range(num_steps):
			out = self.compute_EVB_vector(goal_states, replay_seq, prospective, verbose, EVB_mode, alpha=alpha)
			if verbose:
				MEVBs, (state_need, transition_need, gain) = out
				state_needs[step, :, :, :] = state_need
				transition_needs[step, :, :] = transition_need
				gains[step, :, :] = gain
				all_MEVBs[step, :, :] = MEVBs
			else:
				MEVBs = out

			# Compute the dynamic EVB, using the identity in the function docstring
			DEVBs = np.dot(weights, MEVBs)

			# Log
			if verbose:
				all_DEVBs[step, :] = DEVBs

			# Pick the best one
			if learn_seq:
				best_memory = learn_seq[step]
			else:
				best_memories = np.argwhere(np.abs(DEVBs - np.max(DEVBs)) <= otol).flatten()
				best_memory = self.memory[np.random.choice(best_memories)]

			replay_seq.append(best_memory)

			# Learn!
			if check_convergence:
				backup, mag_delta = self.nstep_learn(replay_seq, ret_update_mag=True, alpha=alpha)
				if mag_delta <= convergence_thresh:  # Reached convergence
					# Cap the storage data structures, if necessary
					if verbose:
						state_needs = state_needs[:step, :, :, :]
						gains = gains[:step, :, :]
						all_MEVBs = all_MEVBs[:step, :, :]
						all_DEVBs = all_DEVBs[:step, :]
						transition_needs = transition_needs[:step, :, :]

					break
			else:
				backup = self.nstep_learn(replay_seq, alpha=alpha)

			backups.append(backup)

		if verbose:
			return np.array(replay_seq), (state_needs, transition_needs, gains, all_MEVBs, all_DEVBs), backups

	def nstep_learn(self, transition_seq, update_policies=True, ret_update_mag=False, alpha=None):
		"""
			Update GR according to transition sequence. Treat last transition in sequence
			as primary transition.
		"""
		if alpha is None:
			alpha = self.alpha

		dG, opt_subseqs = self.compute_nstep_update(transition_seq[-1], replay_seq=transition_seq[:-1])
		self.G += alpha * dG

		if update_policies:
			for goal in self.goal_states:
				self.policies[goal] = self.derive_policy(goal)

		if ret_update_mag:
			return opt_subseqs, np.sum(alpha * np.abs(dG))
		else:
			return opt_subseqs

	def compute_multistep_EVB(self, transition, goal, policy, replay_seq, curr_state, M, G_p=None, prospective=False,
							  alpha=None):
		"""
		Compute the expected value of GR backup for a particular sequence of transitions
		with respect to a particular goal state. Derivation for the factorization
		EVB = need * gain follows from Mattar & Daw (2018), defining GR analogues
		of Q and V functions.

		Args:
			transition ():
			goal ():
			policy ():
			replay_seq ():
			curr_state ():
			M ():
			G_p ():
			prospective ():

		Returns:
		"""
		# Collect variables
		s_k, a_k, s_kp = transition

		# Compute the need of this transition wrt this goal
		need = self.compute_need(curr_state, s_k, M, prospective)

		# Derivation of this update prioritisation shows that EVB = 0 for the special case where
		# s_k == goal
		if s_k == goal:
			return need, 0, 0

		# Compute gain for this transition (and induced n-step backup)
		gain = self.compute_nstep_gain(transition, replay_seq, goal, policy, G_p=G_p, alpha=alpha)

		# Compute and return EVB + factors
		return need, gain, need * gain

	def compute_sweeping_EVB(self, transition, goal, replay_seq, curr_state, M, dG=None, prospective=False):
		"""
		Compute the expected value of GR backup for a particular sequence of transitions
		with respect to a particular goal state. Derivation for the factorization
		EVB = need * gain follows from Mattar & Daw (2018), defining GR analogues
		of Q and V functions.

		Args:
			transition ():
			goal ():
			replay_seq ():
			curr_state ():
			M ():
			dG ():
			prospective ():

		Returns:
		"""

		# Collect variables
		s_k, a_k, s_kp = transition

		# Compute the need of this transition wrt this goal
		need = self.compute_need(curr_state, s_k, M, prospective)

		# Derivation of this update prioritisation shows that EVB = 0 for the special case where
		# s_k == goal
		if s_k == goal:
			return need, 0, 0

		# Compute gain for this transition (and induced n-step backup)
		sweep_gain = self.compute_nstep_Bellman_resid(transition, replay_seq, goal, dG=dG)

		# Compute and return EVB + factors
		return need, sweep_gain, need * sweep_gain

	def get_optimal_subseq(self, replay_seq, goal, tol=1e-6, end=None):
		"""
			Compute the longest subsequence (starting from the end) in replay_seq that constitutes
			an optimal path towards goal under the given policy.
		"""
		if end == goal:  # Special case
			return []

		optimal_subseq = []
		for tdx, transition in enumerate(reversed(replay_seq)):
			s_k, a_k, s_kp = transition

			if tdx == 0 and s_kp != end:  # We require that the sequence conclude at state end
				break

			if s_k == goal:  # Self-trajectories from the goal, to the goal, are somewhat ill-defined
				break

			# If a_k is optimal in s_k...
			if self.check_optimal(s_k, a_k, goal, tol=tol):
				# ... and also, it leads to the first member of the optimal subsequence...
				if not optimal_subseq or s_kp == optimal_subseq[0][0]:
					# then add it to the optimal subsequence.
					optimal_subseq.insert(0, transition)
				else:
					break

			# Otherwise, quit
			else:
				break

		return optimal_subseq

	def get_optimal_stoch_subseq(self, replay_seq, goal, tol=1e-6, end=None):
		"""
		Compute the longest subsequence (starting from the end) in replay_seq that constitutes an optimal decision-
		making sequence towards goal.

		Args:
			replay_seq (list): List of state-action pairs (2-tuples of ints).
			goal (int): Goal state.
			tol (float): Tolerance for optimality.
			end (int): We require that the sequence end at this state.

		Returns:
			optimal_subseq (list): The longest optimal suffix of replay_seq, with respect to goal.
		"""
		if end == goal:  # Special case
			return []

		optimal_subseq = []
		for sdx, sa in enumerate(reversed(replay_seq)):
			s_k, a_k = sa

			if sdx == 0 and self.T[s_k, a_k, end] == 0:  # We require that the sequence have nonzero mass at state `end`
				break

			if s_k == goal:  # Self-trajectories from the goal, to the goal, are somewhat ill-defined
				break

			# If a_k is optimal in s_k...
			if self.check_optimal(s_k, a_k, goal, tol=tol):
				# ... and also, it can lead to the first member of the optimal subsequence...
				if not optimal_subseq or self.T[s_k, a_k, optimal_subseq[0][0]] >= 0:
					# then add it to the optimal subsequence.
					optimal_subseq.insert(0, sa)
				else:
					break

			# Otherwise, quit
			else:
				break

		return optimal_subseq

	def compute_nstep_update(self, transition, replay_seq=None, optimal_subseqs=None, goal_states=None):
		"""
			Given a primary transition and a potentially-empty subsequence of transitions leading to it,
			compute what the net update to the GR is.

			Either one of replay_seq or optimal_subseq must be provided.
		"""
		# Collect variables
		s_k, a_k, s_kp = transition
		dG = np.zeros_like(self.G)

		if goal_states is None:
			goal_states = self.goal_states

		# For each goal...
		computed_subseqs = {}
		for gdx, goal in enumerate(goal_states):
			# Compute GR delta wrt this goal
			if s_kp == goal:
				GR_delta = 1 - self.G[s_k, a_k, goal]
			else:
				GR_delta = self.gamma * np.max(self.G[s_kp, :, goal]) - self.G[s_k, a_k, goal]

			# Implement delta due to primary transition
			dG[s_k, a_k, goal] += GR_delta

			# Find optimal subsequence wrt this goal
			if optimal_subseqs is not None:
				optimal_subseq = optimal_subseqs[gdx]
			elif not self.check_optimal(s_k, a_k, goal):  # Exploratory actions do not backpropagate
				optimal_subseq = []
			else:
				optimal_subseq = self.get_optimal_subseq(replay_seq, goal, end=s_k)

			computed_subseqs[goal] = optimal_subseq

			# Backpropagate delta throughout this subsequence as relevant
			for mdx, memory in enumerate(reversed(optimal_subseq)):
				s_m, a_m, s_mp = memory
				dG[s_m, a_m, goal] += (self.gamma ** (mdx + 1)) * GR_delta

		return dG, computed_subseqs

	def compute_stoch_nstep_update(self, sa, replay_seq=None, optimal_subseqs=None, goal_states=None):
		"""
		Given a state-action pair and a potentially-empty list of state-action pairs leading to it,
		compute the net update to the GR. Unlike compute_nstep_update(), this function does not assume a particular
		successor state but instead performs the full update over all possible successors, weighted by the MDP
		transition matrix. This is needed in stochastic regimes where state-action pairs do not define a unique
		successor, but instead define a distribution over possible successor states. (An alternative is to sample
		a successor, but compute_nstep_update() can be used in that case.)

		Args:
			sa (tuple): Tuple of ints representing the state-action pair (state, action).
			replay_seq (list): List of previous chosen replays.
			optimal_subseqs (list): List containing the most optimal subsequence of state-action pairs leading to sa.
			goal_states (list): The goal states with respect to which the update is computed.

		Returns:
			dG (np.ndarray): The update to the agent's Geodesic representation.
			computed_subseqs (dict): For each goal, the optimal subsequences chosen.

		TODO: add sampling flag, support for sampled successor states
		TODO: think about whether the backprop scaling term makes sense (probably yes)
		"""
		# Collect variables
		s_k, a_k = sa
		dG = np.zeros_like(self.G)

		if goal_states is None:
			goal_states = self.goal_states

		# For each goal...
		computed_subseqs = {}
		for gdx, goal in enumerate(goal_states):
			# Compute GR delta wrt this goal (full backup for stochastic environments)
			GR_delta = 0
			for s_kp in range(self.num_states):
				if s_kp == goal:
					GR_delta += self.T[s_k, a_k, goal] * (1 - self.G[s_k, a_k, goal])
				else:
					GR_delta += self.T[s_k, a_k, s_kp] * (self.gamma * np.max(self.G[s_kp, :, goal])
														  - self.G[s_k, a_k, goal])

			# Implement delta due to primary transition
			dG[s_k, a_k, goal] += GR_delta

			# Find optimal subsequence wrt this goal
			if optimal_subseqs is not None:
				optimal_subseq = optimal_subseqs[gdx]
			elif not self.check_optimal(s_k, a_k, goal):  # Exploratory actions do not backpropagate
				optimal_subseq = []
			else:
				optimal_subseq = self.get_optimal_subseq(replay_seq, goal, end=s_k)

			computed_subseqs[goal] = optimal_subseq

			# Backpropagate delta throughout this subsequence as relevant
			s_mp = s_k
			for mdx, memory in enumerate(reversed(optimal_subseq)):
				s_m, a_m = memory
				dG[s_m, a_m, goal] += self.T[s_m, a_m, s_mp] * (self.gamma ** (mdx + 1)) * GR_delta
				s_mp = s_m

		return dG, computed_subseqs

	def check_optimal(self, s_k, a_k, goal, tol=1e-6):
		return abs(self.G[s_k, a_k, goal] - np.max(self.G[s_k, :, goal])) <= tol

	def compute_nstep_Bellman_resid(self, transition, replay_seq, goal, dG=None, optimal_subseq=None):
		"""

		Args:
			transition ():
			replay_seq ():
			goal ():
			dG ():
			optimal_subseq ():

		Returns:

		"""

		# Collect variables
		s_k, a_k, s_kp = transition

		# Get optimal subsequence of replay_seq with respect to goal
		if optimal_subseq is None:
			optimal_subseq = self.get_optimal_subseq(replay_seq, goal, end=s_k)

		# Compute new GR given this primary transition + optimal subsequence
		if dG is None:
			dG, _ = self.compute_nstep_update(transition, optimal_subseqs=[optimal_subseq], goal_states=[goal])

		return np.sum(np.abs(dG))

	def compute_nstep_gain(self, transition, replay_seq, goal, policy, G_p=None, optimal_subseq=None, alpha=None):
		"""
			Compute gain blah
		"""
		if alpha is None:
			alpha = self.alpha

		# Collect variables
		s_k, a_k, s_kp = transition

		# Get optimal subsequence of replay_seq with respect to goal
		if optimal_subseq is None:
			optimal_subseq = self.get_optimal_subseq(replay_seq, goal, end=s_k)

		# Compute new GR given this primary transition + optimal subsequence
		if G_p is None:
			dG, _ = self.compute_nstep_update(transition, optimal_subseqs=[optimal_subseq], goal_states=[goal])
			G_p = self.G.copy() + alpha * dG

		## Compute gain
		gain = 0

		# Get gain due to primary transition
		pi_p = self.update_state_policy(s_k, goal, G=G_p)
		for action in range(self.num_actions):
			gain += (pi_p[s_k, action] - policy[s_k, action]) * G_p[s_k, action, goal]

		# Get gain due to states visited during n-step backup
		for mdx, memory in enumerate(optimal_subseq):
			s_m, a_m, s_mp = memory
			pi_p = self.update_state_policy(s_m, goal, G=G_p)
			for action in range(self.num_actions):
				gain += (pi_p[s_m, action] - policy[s_m, action]) * G_p[s_m, action, goal]

		return gain

	def compute_need(self, state, s_k, M, prospective=False):
		"""
			Compute the need term of the GR EVB equation.
		"""

		if prospective:  # Average needs across all possible start states
			return np.average(M[:, s_k], weights=self.s0_dist)
		else:
			return M[state, s_k]

	def compute_occupancy(self, policy, T):
		"""
			Compute future state occupancy matrix given a policy `policy`
			and transition dynamics matrix `T`
		"""
		# Convert dynamics + policy to one-step transition matrix
		one_step_T = dynamics_policy_onestep(policy, T)

		# Compute resultant future occupancy matrix and evaluate
		M = np.linalg.inv(np.eye(self.num_states) - self.gamma * one_step_T)

		return M


class SftmxGeodesicAgent(GeodesicAgent):
	def __init__(self, num_states: int, num_actions: int, goal_states: np.ndarray,
				 T: np.ndarray,
				 goal_dist: np.ndarray = None,
				 s0_dist: np.ndarray = None,
				 alpha: float = 0.3,
				 gamma: float = 0.95,
				 min_gain: float = 0,
				 policy_temperature: float = 1.0):
		"""

		Args:
			policy_temperature:
		"""

		self.policy_temperature = policy_temperature
		super().__init__(num_states, num_actions, goal_states, T, goal_dist, s0_dist,
						 alpha, gamma, min_gain)

	def derive_policy(self, goal_state, G=None, set_policy=False, epsilon=0, policy_temperature=None):
		"""
		Derive the policy for reaching a given goal state. Unlike the absolute max in the vanilla
		GeodesicAgent class, here we use a softmax. (As such, the softmax parameter is ignored.)

		Args:
			goal_state (int): The goal with respect to which the policy is derived.
			G (np.ndarray, optional): The Geodesic representation over which the policy is derived.
				If none is provided, the agent's current one will be used.
			set_policy (boolean): If True, the computed policy will update the agent's current policy
				 for the specified goal.
			epsilon (float): Epsilon parameter for epsilon-greedy action policy.
			policy_temperature (float):

		Returns:
			policy (np.ndarray): The computed policy.
		"""

		# Allow arbitrary G to be fed in, default to current G
		if G is None:
			G = self.G

		# Ditto the policy temperature
		if policy_temperature is None:
			policy_temperature = self.policy_temperature

		policy = np.zeros((self.num_states, self.num_actions))

		# Compute policy
		for state in range(self.num_states):
			# Recall that the GR is actually an SR and that SR values are actually Q-values
			# from a certain point of view. So we can feed them directly into the softmax equation.
			probs = softmax(G[state, :, goal_state], policy_temperature)
			policy[state, :] = probs

		# Cache if wanted
		if set_policy:
			self.policies[goal_state] = policy

		return policy

	def update_state_policy(self, state, goal_state, G=None, set_policy=False, epsilon=0, policy_temperature=None):
		"""
		Update the internal policy only for a given state. Unlike the vanilla GeodesicAgent
		class, here we use a softmax to compute the best policy.

		Args:
			state (int): The state receiving the policy update
			goal_state (int): The state with respect to which the policy is being updated
			G (np.ndarray, optional): The Geodesic representation over which the policy is derived.
				If none is provided, the agent's current one will be used.
			set_policy (boolean): If True, the computed policy will update the agent's current policy
				 for the specified goal.
			epsilon (float): Epsilon parameter for epsilon-greedy action policy.

		Returns:
			policy (np.ndarray): The computed policy.
		"""

		# Allow arbitrary G to be fed in, default to current G
		if G is None:
			G = self.G
		if policy_temperature is None:
			policy_temperature = self.policy_temperature

		# Compute policy
		probs = softmax(G[state, :, goal_state], policy_temperature)

		if not set_policy:  # Only re-copy the whole thing if we're not planning on saving it.
			policy = self.policies[goal_state].copy()
		else:
			policy = self.policies[goal_state]

		policy[state, :] = probs

		# Cache if wanted
		if set_policy:
			self.policies[goal_state] = policy

		return policy

if __name__ == '__main__':
	pass
