import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import math

from MarkovDecisionProcess import MarkovDecisionProcess
from RL_utils import oned_twod, compute_occupancy, twod_oned


class GridWorld(MarkovDecisionProcess):
	"""
	The GridWorld class is a particular instance of the MarkovDecisionProblem
	class. In particular, it is characterized by a 2d state space. It corresponds
	to a "grid world" in which states are locations. Certain transitions in space
	may be disallowed, corresponding to the existence of barriers.
	"""

	def __init__(self, width, height, epsilon=0, walls=None, term_states=None, init_state_distribution=None,
				 semipermeable=False):
		"""
		Initialize GridWorld object with given dimensions, 4 actions (compass directions)
		and given set of barriers ((S,A) tuples that are banned). Each action
		will lead to the corresponding target state with probability (1 - epsilon)
		and to any one of the neighbours uniformly otherwise.

		Action key:
			0: left
			1: up
			2: right
			3: down

		Args:
			width (int):
			height (int):
			epsilon (float): Dynamics noise term.
			walls ():
			term_states (list):
			init_state_distribution (np.ndarray):
		"""

		if walls is None:
			walls = []
		else:
			walls = walls.copy()

		if term_states is None:
			term_states = []

		self.width = width
		self.height = height

		num_states = width * height
		num_actions = 4

		## Prepare list of blocked paths, including user-defined walls and also
		## arena boundaries
		self.blocked_paths = []

		self.arena_boundaries = []
		# Top and bottom rows can't move up or down, respectively
		for i in range(self.width):
			self.arena_boundaries.append((i, 1))  # Top row can't move up
			self.arena_boundaries.append((num_states - self.width + i, 3))  # Bottom row can't move down

		# Left and right columns can't move left or right, respectively
		for i in range(self.height):
			self.arena_boundaries.append((i * self.width, 0))  # Left column can't move left
			self.arena_boundaries.append((self.width - 1 + (i * self.width), 2))  # Right column can't move right

		self.blocked_paths.extend(self.arena_boundaries)

		# Add user-defined walls
		if not semipermeable:
			ds_walls = self.double_side_walls(walls)
			walls.extend(ds_walls)

		self.blocked_paths.extend(walls)

		# Build transition matrix
		transitions = np.zeros((num_states, num_actions, num_states))
		for src in range(num_states):
			if src in term_states:
				transitions[src, :, src] = 1
			else:
				for action in range(num_actions):
					transitions[src, action, :] = self.__get_succ_dist(src, action, epsilon, self.blocked_paths)

		self.transitions = transitions

		# Having all these things, instantiate the underlying MDP
		super().__init__(self.transitions, num_actions, init_state_distribution)

	def double_side_walls(self, walls):
		"""
		For each wall in `walls`, add the complementary wall to make it block motion in both directions. No checks
		are performed that the walls in `walls` are proper or valid.

		Args:
			walls (list): List of banned (state, action) tuples.

		Returns:

		"""
		if not walls:
			return []

		double_walls = []
		for wall in walls:
			if wall in self.arena_boundaries:  # Can't be double-sided (no states exist on the other side)
				continue

			double_walls.append(self.get_double_wall(wall))

		return double_walls

	def get_double_wall(self, wall):
		s, a = wall

		if a == 0:
			return s - 1, 2
		elif a == 1:
			return s - self.width, 3
		elif a == 2:
			return s + 1, 0
		elif a == 3:
			return s + self.width, 1

	def __get_succ_dist(self, src, action, epsilon, blocked_paths):
		"""
			Given a source state, an action performed at that state,
			as well as a list of disallowed state-action pairs,
			return the successor state distribution. If the action is allowed,
			it will proceed to the indicated state with probability 1 - epsilon,
			otherwise it will move to one of its neighbours uniformly. If that neighbour
			is blocked, it will stay in the current state.
		"""
		target_dist = np.zeros(self.width * self.height)

		## Assign 1 - epsilon based on action
		# If action is blocked, stay
		if (src, action) in blocked_paths:
			target_dist[src] += 1 - epsilon
		# Otherwise...
		else:
			if action == 0:  # go left
				target_dist[src - 1] += 1 - epsilon
			if action == 1:  # go up
				target_dist[src - self.width] += 1 - epsilon
			if action == 2:  # go right
				target_dist[src + 1] += 1 - epsilon
			if action == 3:  # go down
				target_dist[src + self.width] += 1 - epsilon

		# Movement due to epsilon noise
		poss_neighbours = np.ones(4)  # One for each cardinal direction
		if (src, 0) in blocked_paths:  # Can't move left
			poss_neighbours[0] = 0
		if (src, 1) in blocked_paths:  # Can't move up
			poss_neighbours[1] = 0
		if (src, 2) in blocked_paths:  # Can't move right
			poss_neighbours[2] = 0
		if (src, 3) in blocked_paths:  # Can't move down
			poss_neighbours[3] = 0

		# epsilon probability to stay for each blocked direction
		target_dist[src] += (4 - np.sum(poss_neighbours)) * (epsilon / 4)

		# epsilon / 4 probability to all the available neighbours
		if poss_neighbours[0] == 1:  # left is open
			target_dist[src - 1] += epsilon / 4
		if poss_neighbours[1] == 1:  # up is open
			target_dist[src - self.width] += epsilon / 4
		if poss_neighbours[2] == 1:  # right is open
			target_dist[src + 1] += epsilon / 4
		if poss_neighbours[3] == 1:  # down is open
			target_dist[src + self.width] += epsilon / 4

		return target_dist

	def set_terminal(self, states):
		"""
		Set terminal states.

		Args:
			states (list): List of states to be made terminal
		"""

		for state in states:
			self.transitions[state, :, :] = 0  # This state can't lead anywhere...
			self.transitions[state, :, state] = 1  # ... but to itself

	def draw(self, use_reachability=False, ax=None, figsize=(12, 12), title='', label_states=False,
             wall_width=2, return_unreachable=False):
		# Generic setup
		if not ax:
			fig, ax = plt.subplots(1, 1, figsize=figsize)
		
		ax.set_xlim(-0.5, self.width + 0.5)
		ax.set_ylim(self.height + 0.5, -0.5)  # Puts origin at top left
		ax.xaxis.tick_top()  # Puts x-axis ticks on top

		# Draw grid lines manually, because dealing with major and minor ticks in plt.grid() is a pain
		for y in range(self.height):
			ax.plot([0, self.width], [y, y], alpha=0.35, color='gray', linewidth=0.5)

		for x in range(self.width):
			ax.plot([x, x], [0, self.height], alpha=0.35, color='gray', linewidth=0.5)

		# Draw walls
		for wall in self.blocked_paths:
			state, action = wall
			row, col = oned_twod(state, self.width, self.height)

			# check if wall is on the boundary
			if (row == 0 and action == 1) or (row == self.height - 1 and action == 3) or (col == 0 and action == 0) or (col == self.width - 1 and action == 2):
				lw = 1
			else:
				lw = wall_width

			if action == 0:  # Block left wall
				ax.plot([col, col], [row, row + 1], color='k', linewidth=lw)
			elif action == 1:  # Block top wall
				ax.plot([col, col + 1], [row, row], color='k', linewidth=lw)
			elif action == 2:  # Block right wall
				ax.plot([col + 1, col + 1], [row, row + 1], color='k', linewidth=lw)
			else:  # Block bottom wall
				ax.plot([col, col + 1], [row + 1, row + 1], color='k', linewidth=lw)

		# Shade in unreachable cells
		unreachable_states = None
		if use_reachability:
			uniform_policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
			occupancy = compute_occupancy(uniform_policy, self.transitions)
			reachable = np.zeros((self.num_states, self.num_states))
			for start_state in range(self.num_states):
				# Ignore contributions from non-starting states
				if math.isclose(self.s0_dist[start_state], 0): 
					continue

				# If this state is reachable, mark it so
				for target_state in range(self.num_states):
					if not math.isclose(occupancy[start_state, target_state], 0):
						reachable[start_state, target_state] = 1

			# Shade in states that have corresponding columns all equal to 0 (not reachable from anywhere)
			unreachable_states = np.where(~reachable.any(axis=0))[0]
			for state in unreachable_states:
				row, col = oned_twod(state, self.width, self.height)
				rect = patches.Rectangle((col, row), 1, 1, facecolor='k')
				ax.add_patch(rect)

		# Give a title
		ax.set_title(title)

		# If desired, label every state
		if label_states:
			for x in range(self.width):
				for y in range(self.height):
					state_id = twod_oned(x, y, self.width)
					ax.text(x + 0.5, y + 0.5, '%d' % state_id)

		if return_unreachable:
			return ax, unreachable_states
		else:
			return ax

	def generate_adjacency_matrix(self, allow_self_loops=True, allow_multiloops=True):
		A = np.zeros((self.num_states, self.num_states))
		for i in range(self.num_states):
			for j in range(self.num_states):
				if not allow_self_loops and i == j:
					continue

				for a in range(self.num_actions):
					if self.transitions[i, a, j] > 0:
						if allow_multiloops:
							A[i, j] += 1
						else:
							A[i, j] = 1

		return A




class Arena(GridWorld):
	"""
		Specific class for an arena GridWorld -- just an open rectangle. Defaults to the start position
		in the northwest corner, and deterministic dynamics.
	"""

	def __init__(self, width, height, stoch=0, init_state_distribution=None):
		if init_state_distribution is None:
			init_state_distribution = np.zeros(width * height)
			init_state_distribution[0] = 1

		super().__init__(width, height, epsilon=stoch, init_state_distribution=init_state_distribution)


class Bottleneck(GridWorld):
	"""
		Specific class for a bottleneck GridWorld -- two arenas connected by a corridor. Defaults the start
		position to the northwest corner, with deterministic dynamics.
	"""

	def __init__(self, room_width, corridor_width, height, stoch=0, init_state_distribution=None):
		full_width = 2 * room_width + corridor_width
		self.room_width = room_width
		self.corridor_width = corridor_width

		if init_state_distribution is None:
			init_state_distribution = np.zeros(full_width * height)
			init_state_distribution[0] = 1

		# Build list of inaccessible states due to walls
		corr_row = int(height // 2)  # Integer division, hopefully
		left_col = int(room_width - 1)
		right_col = int(room_width + corridor_width)

		banned_states = []
		for row in range(height):
			if row == corr_row:
				continue

			banned_states.extend([i for i in range(row * full_width + left_col + 1, row * full_width + right_col)])

		self.banned_states = banned_states

		# Get walls to cut off irrelevant GridWorld sections
		add_walls = self.__bottleneck_walls(room_width, corridor_width, height)
		super().__init__(full_width, height, epsilon=stoch, walls=add_walls, init_state_distribution=init_state_distribution)

	@staticmethod
	def __bottleneck_walls(room_width, corridor_width, height):
		"""
			Build the wall list for bottleneck enclosures. Actions 0-4 correspond to
				0: west
				1: north
				2: east
				3: south

			Note that these walls, as written, are semipermeable. If the agent were to somehow
			start in the blocked-off areas, they would be able to enter the main area.
		"""
		corr_row = int(height // 2)  # Integer division, hopefully
		left_col = int(room_width - 1)
		right_col = int(room_width + corridor_width)
		full_width = 2 * room_width + corridor_width

		walls = []

		# Add top and bottom walls to corridor states
		for x in range(left_col + 1, right_col):
			state_id = corr_row * full_width + x
			walls.append((state_id, 1))  # Can't go up
			walls.append((state_id, 3))  # Can't go down

		# Add bottom walls to states above corridor
		for x in range(left_col + 1, right_col):
			state_id = (corr_row - 1) * full_width + x
			walls.append((state_id, 3))  # Can't go down

		# Add top walls to states below corridor
		for x in range(left_col + 1, right_col):
			state_id = (corr_row + 1) * full_width + x
			walls.append((state_id, 1))  # Can't go up

		# Add right walls to left column states
		for y in range(height):
			if y == corr_row:  # No walls on corridor entrance
				continue

			state_id = y * full_width + left_col
			walls.append((state_id, 2))  # Can't go right

		# Add left walls to states to the right of left column
		for y in range(height):
			if y == corr_row:  # No walls on corridor entrance
				continue

			state_id = y * full_width + (left_col + 1)
			walls.append((state_id, 0))  # Can't go left

		# Add left walls to right column states
		for y in range(height):
			if y == corr_row:  # No walls on corridor entrance
				continue

			state_id = y * full_width + right_col
			walls.append((state_id, 0))  # Can't go left

		# Add right walls to states to the left of right column
		for y in range(height):
			if y == corr_row:  # No walls on corridor entrance
				continue

			state_id = y * full_width + (right_col - 1)
			walls.append((state_id, 2))  # Can't go right

		return walls

	def get_all_transitions(self, tol=1e-6, filter_self_transitions=True):
		"""
			Return a list of all state-action pairs and their successors. Samples only once,
			so only complete for grid-stochasticity = 0.
		"""
		all_transitions = super().get_all_transitions(tol=tol, filter_self_transitions=filter_self_transitions)
		return [transition for transition in all_transitions if transition[0] not in self.banned_states]

	@staticmethod
	def get_valid_states(room_width, corridor_width, height):
		corr_row = int(height // 2)  # Integer division, hopefully
		width = 2 * room_width + corridor_width

		# Build valid states
		valid_states = []

		# States above tunnel
		for i in range(height):
			# Tunnel
			if i == corr_row:
				all_states = np.arange(i * width, (i + 1) * width)
				valid_states.extend(list(all_states))
			else:
				# Left room
				left_states = np.arange(i * width, i * width + room_width)
				valid_states.extend(list(left_states))

				# Right room
				start = (i * width) + room_width + corridor_width
				end = (i + 1) * width
				right_states = np.arange(start, end)
				valid_states.extend(list(right_states))

		return valid_states


class LinearChamber(GridWorld):
	"""
		Specific class for a linear chamber GridWorld -- just a height 1 rectangle. Defaults the start position
		to the western edge, and dterministic dynamics.
	"""
	def __init__(self, length, stoch=0, init_state_distribution=None):
		if init_state_distribution is None:
			init_state_distribution = np.zeros(length)
			init_state_distribution[0] = 1

		super().__init__(length, height=1, epsilon=stoch, init_state_distribution=init_state_distribution)


if __name__ == '__main__':
	pass
	