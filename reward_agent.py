import numpy as np
from RL_utils import dynamics_policy_onestep, softmax, compute_occupancy

class RewardAgent(object):
    def __init__(self, num_states: int, num_actions: int,
                 T: np.ndarray,
                 s0_dist: np.ndarray = None,
                 alpha: float = 0.3,
                 gamma: float = 0.95,
                 min_gain: float = 0):

        # MDP properties
        self.num_states = num_states
        self.num_actions = num_actions
        self.s0_dist = s0_dist
        self.T = T
        self.gamma = gamma

        # Agent properties
        self.min_gain = min_gain
        self.alpha = alpha

        self.Q = np.zeros((num_states, num_actions))
        self.memory = []

        uniform_policy = np.ones((num_states, num_actions)) / num_actions
        self.policy = uniform_policy

    def value_iteration(self, num_iters=10000, reward_vec=None):
        if reward_vec is None:
            reward_vec = np.zeros(self.num_states)

        true_V = np.zeros(self.num_states)
        for _ in range(num_iters):
            for s in range(self.num_states):
                sums = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    for sp in range(self.num_states):
                        sums[a] += self.T[s, a, sp] * (reward_vec[sp] + self.gamma * true_V[sp])

                true_V[s] = np.max(sums)

        return true_V




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

    def basic_learn(self, transition, decay_rate=None, noise=0, update_policy=True, alpha=None):
        """
            One-step transition learning, with forgetting.

            Args:
                transition ():
                decay_rate ():
                noise ():
                update_policies():
        """
        if decay_rate is None:
            decay_rate = 1  # No forgetting
        if alpha is None:
            alpha = self.alpha

        s_k, a_k, s_kp, r_k = transition
        dQ = np.zeros_like(self.Q)  # Additive change to Q (reward/value)
        mQ = np.ones_like(self.Q) * decay_rate  # Autoregressive multiplicative change to Q

        dQ[s_k, a_k] = r_k + self.gamma * np.max(self.Q[s_kp, :])
        mQ[s_k, a_k] = (1 - alpha)

        # Update Q according to interpolation form of the update equation
        self.Q = mQ * self.Q + alpha * dQ

        # Potentially add noise
        if noise > 0:
            self.Q += np.random.normal(0, noise, size=self.Q.shape)

        # Consolidate policies according to updated GR
        if update_policy:
            self.policy = self.derive_policy()

    def derive_policy(self, Q=None, epsilon=0, set_policy=True):
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
        if Q is None:
            Q = self.Q

        policy = np.zeros((self.num_states, self.num_actions))

        # Compute policy
        for state in range(self.num_states):
            best_actions = np.flatnonzero(Q[state, :] == np.max(Q[state, :]))
            num_best_actions = len(best_actions)  # Split 1 - epsilon ties equally

            policy[state, :] = epsilon / self.num_actions
            policy[state, best_actions] += (1 - epsilon) / num_best_actions  # Deterministic for epsilon = 0

        # Cache if wanted
        if set_policy:
            self.policy = policy

        return policy

    def compute_EVB_vector(self, replay_seq, prospective=False, verbose=False, alpha=None):
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

        EVBs = np.zeros(len(self.memory))  # Best transition is picked greedily at each step

        state_needs = None
        transition_needs = None
        gains = None
        if verbose:
            state_needs = np.zeros((self.num_states, self.num_states))
            transition_needs = np.zeros((len(self.memory)))
            gains = np.zeros(len(self.memory))

        # Compute EVB for all transitions across all goal states
        M_pi = compute_occupancy(self.policy, self.T)

        # Log, if wanted
        if verbose:
            state_needs = M_pi

        # Compute EVB for each transition
        for tdx, transition in enumerate(self.memory):
            dQ, _ = self.compute_nstep_update(transition, replay_seq=replay_seq)

            Q_p = self.Q + alpha * dQ
            need, gain, evb = self.compute_multistep_EVB(transition, self.policy,
                                                         replay_seq,
                                                         curr_state=self.curr_state,
                                                         M=M_pi,
                                                         Q_p=Q_p,
                                                         prospective=prospective)

            EVBs[tdx] = evb

            # Log quantities, if desired
            if verbose:
                gains[tdx] = gain
                transition_needs[tdx] = need

        if verbose:
            return EVBs, (state_needs, transition_needs, gains)
        else:
            return EVBs

    def check_optimal(self, s_k, a_k, tol=1e-6):
        return abs(self.Q[s_k, a_k] - np.max(self.Q[s_k, :])) <= tol

    def compute_nstep_update(self, transition, replay_seq=None, optimal_subseq=None):
        """
            Given a primary transition and a potentially-empty subsequence of transitions leading to it,
            compute what the net update to the GR is.

            Either one of replay_seq or optimal_subseq must be provided.
        """
        # Collect variables
        s_k, a_k, s_kp, r = transition
        dQ = np.zeros_like(self.Q)

        # For each goal...
        Q_delta = r + self.gamma * np.max(self.Q[s_kp, :]) - self.Q[s_k, a_k]
        dQ[s_k, a_k] += Q_delta

        # Find optimal subsequence wrt this goal
        if optimal_subseq is None and self.check_optimal(s_k, a_k):  # Exploratory actions do not backpropagate
            optimal_subseq = self.get_optimal_subseq(replay_seq, end=s_k)
        elif optimal_subseq is None:
            optimal_subseq = []

        # Backpropagate delta throughout this subsequence as relevant
        for mdx, memory in enumerate(reversed(optimal_subseq)):
            s_m, a_m, s_mp, r = memory
            dQ[s_m, a_m] += (self.gamma ** (mdx + 1)) * Q_delta

        return dQ, optimal_subseq

    def nstep_learn(self, transition_seq, update_policy=True, ret_update_mag=False, alpha=None):
        """
            Update GR according to transition sequence. Treat last transition in sequence
            as primary transition.
        """
        if alpha is None:
            alpha = self.alpha

        dQ, opt_subseq = self.compute_nstep_update(transition_seq[-1], replay_seq=transition_seq[:-1])
        self.Q += alpha * dQ

        if update_policy:
            self.policy = self.derive_policy()

        if ret_update_mag:
            return opt_subseq, np.sum(alpha * np.abs(dQ))
        else:
            return opt_subseq

    def get_optimal_subseq(self, replay_seq, tol=1e-6, end=None):
        """
            Compute the longest subsequence (starting from the end) in replay_seq that constitutes
            an optimal path towards goal under the given policy.
        """
        optimal_subseq = []
        for tdx, transition in enumerate(reversed(replay_seq)):
            s_k, a_k, s_kp, r = transition

            if tdx == 0 and s_kp != end:  # We require that the sequence conclude at state end
                break

            # If a_k is optimal in s_k...
            if self.check_optimal(s_k, a_k, tol=tol):
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

    def compute_multistep_EVB(self, transition, policy, replay_seq, curr_state, M, Q_p=None, prospective=False):
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
        s_k, a_k, s_kp, r_k = transition

        # Compute the need of this transition wrt this goal
        need = self.compute_need(curr_state, s_k, M, prospective)

        # Compute gain for this transition (and induced n-step backup)
        gain = self.compute_nstep_gain(transition, replay_seq, policy, Q_p=Q_p)

        # Compute and return EVB + factors
        return need, gain, need * gain

    def compute_need(self, state, s_k, M, prospective=False):
        """
            Compute the need term of the GR EVB equation.
        """

        if prospective:  # Average needs across all possible start states
            return np.average(M[:, s_k], weights=self.s0_dist)
        else:
            return M[state, s_k]

    def compute_nstep_gain(self, transition, replay_seq, policy, Q_p=None, optimal_subseq=None, alpha=None):
        """
            Compute gain blah
        """
        if alpha is None:
            alpha = self.alpha

        # Collect variables
        s_k, a_k, s_kp, r_k = transition

        # Get optimal subsequence of replay_seq with respect to goal
        if optimal_subseq is None:
            optimal_subseq = self.get_optimal_subseq(replay_seq, end=s_k)

        # Compute new GR given this primary transition + optimal subsequence
        if Q_p is None:
            dQ, _ = self.compute_nstep_update(transition, optimal_subseqs=[optimal_subseq])
            Q_p = self.Q.copy() + alpha * dQ

        ## Compute gain
        gain = 0

        # Get gain due to primary transition
        pi_p = self.update_state_policy(s_k, Q=Q_p)
        for action in range(self.num_actions):
            gain += (pi_p[s_k, action] - policy[s_k, action]) * Q_p[s_k, action]

        # Get gain due to states visited during n-step backup
        for mdx, memory in enumerate(optimal_subseq):
            s_m, a_m, s_mp, r_m = memory
            pi_p = self.update_state_policy(s_m, Q=Q_p)
            for action in range(self.num_actions):
                gain += (pi_p[s_m, action] - policy[s_m, action]) * Q_p[s_m, action]

        return gain

    def update_state_policy(self, state, Q=None, set_policy=False, epsilon=0):
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
        if Q is None:
            Q = self.Q

        # Compute policy
        best_actions = np.flatnonzero(Q[state, :] == np.max(Q[state, :]))
        num_best_actions = len(best_actions)  # Split 1 - epsilon ties equally

        if not set_policy:  # Only re-copy the whole thing if we're not planning on saving it.
            policy = self.policy.copy()
        else:
            policy = self.policy

        policy[state, :] = epsilon / self.num_actions  # Deterministic for epsilon = 0
        policy[state, best_actions] += (1 - epsilon) / num_best_actions

        # Cache if wanted
        if set_policy:
            self.policy = policy

        return policy

    def replay(self, num_steps, prospective=False, verbose=False, alpha=None, otol=1e-6, conv_thresh=0.0):
        """

        Args:
            num_steps:
            prospective:
            verbose:
            alpha:

        Returns:

        """
        # If verbose usage, build storage structures
        state_needs = None
        transition_needs = None
        gains = None
        all_MEVBs = None
        if verbose:
            state_needs = np.zeros((num_steps, self.num_states, self.num_states))
            transition_needs = np.zeros((num_steps, len(self.memory)))
            gains = np.zeros((num_steps, len(self.memory)))
            all_MEVBs = np.zeros((num_steps, len(self.memory)))

        # Start replaying
        replay_seq = []  # Maintain a list of replayed memories for use in multistep backups
        backups = []  # Maintain a list of transitions replayed in each backup step
        for step in range(num_steps):
            out = self.compute_EVB_vector(replay_seq, prospective, verbose, alpha=alpha)
            if verbose:
                MEVBs, (state_need, transition_need, gain) = out
                state_needs[step, :, :] = state_need
                transition_needs[step, :] = transition_need
                gains[step, :] = gain
                all_MEVBs[step, :] = MEVBs
            else:
                MEVBs = out

            # Check for convergence
            if conv_thresh > 0.0 and np.max(MEVBs) < conv_thresh:
                return np.array(replay_seq), (state_needs, transition_needs, gains, all_MEVBs), backups

            # Pick the best one
            best_memories = np.argwhere(np.abs(MEVBs - np.max(MEVBs)) <= otol).flatten()
            best_memory = self.memory[np.random.choice(best_memories)]

            replay_seq.append(best_memory)

            # Learn!
            backup = self.nstep_learn(replay_seq, alpha=alpha)
            backups.append(backup)

        if verbose:
            return np.array(replay_seq), (state_needs, transition_needs, gains, all_MEVBs), backups

class SftmxRewardAgent(RewardAgent):
    def __init__(self, num_states: int, num_actions: int,
                 T: np.ndarray,
                 s0_dist: np.ndarray = None,
                 alpha: float = 0.3,
                 gamma: float = 0.95,
                 min_gain: float = 0,
                 policy_temperature: float = 0.3):
        self.policy_temperature = policy_temperature
        super().__init__(num_states, num_actions, T, s0_dist,
                         alpha, gamma, min_gain)

    def derive_policy(self, Q=None, set_policy=False, epsilon=0, policy_temperature=None):
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
        if Q is None:
            Q = self.Q

        # Ditto the policy temperature
        if policy_temperature is None:
            policy_temperature = self.policy_temperature

        policy = np.zeros((self.num_states, self.num_actions))

        # Compute policy
        for state in range(self.num_states):
            # Recall that the GR is actually an SR and that SR values are actually Q-values
            # from a certain point of view. So we can feed them directly into the softmax equation.
            probs = softmax(Q[state, :], policy_temperature)
            policy[state, :] = probs

        # Cache if wanted
        if set_policy:
            self.policy = policy

        return policy

    def update_state_policy(self, state, Q=None, set_policy=False, epsilon=0, policy_temperature=None):
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
        if Q is None:
            Q = self.Q
        if policy_temperature is None:
            policy_temperature = self.policy_temperature

        # Compute policy
        probs = softmax(Q[state, :], policy_temperature)

        if not set_policy:  # Only re-copy the whole thing if we're not planning on saving it.
            policy = self.policy.copy()
        else:
            policy = self.policy

        policy[state, :] = probs

        # Cache if wanted
        if set_policy:
            self.policy = policy

        return policy


if __name__ == '__main__':
    from gridworld import Arena
    width = 3
    height = 3
    nact = 4
    nstate = width * height
    rew_state = nstate - 1
    rvec = np.zeros(nstate)
    rvec[rew_state] = 1

    s0_dist = np.zeros(nstate)
    s0_dist[0] = 1

    arena = Arena(width, height)
    arena.set_terminal([rew_state])
    T = arena.transitions
    all_experiences = arena.get_all_transitions()
    exps_with_reward = [(s, a, sp, 1) if sp == rew_state else (s, a, sp, 0) for (s, a, sp) in all_experiences]
    no_redundant_exps = [(s, a, sp, r) for (s, a, sp, r) in exps_with_reward if s != sp]

    ra = RewardAgent(nstate, nact, T, s0_dist, alpha=0.3)
    ra.remember(no_redundant_exps)
    ra.curr_state = 0
    ra.basic_learn((nstate - 2, 2, nstate - 1, 1))
    ra.basic_learn((nstate - 3, 2, nstate - 2, 0))
    ra.basic_learn((0, 3, 3, 0))
    ra.basic_learn((3, 3, 6, 0))
    replays, stats, backups = ra.replay(1, verbose=True, prospective=True)
    #
    # print('done')
    # for i in range(50000):
    #     action, next_state, reward = arena.execute_policy(ra.curr_state, policy=ra.policy, reward_vector=rvec)
    #     ra.basic_learn((ra.curr_state, action, next_state, reward), update_policy=False)
    #     ra.curr_state = next_state
    #
    #     if ra.curr_state == rew_state:
    #         ra.curr_state = 0

    print('done')
