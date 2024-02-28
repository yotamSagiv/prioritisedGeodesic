import numpy as np
from MarkovDecisionProcess import MarkovDecisionProcess


class DiGraph(MarkovDecisionProcess):
	"""
		The DiGraph class is a particular instance of the MarkovDecisionProblem
		class. In particular, it is a directed graph in which environmental
		dynamics are deterministic.
	"""

	def __init__(self, num_vertices, edges, init_state_dist=None):
		"""
		Initialize the DiGraph.

		Args:
			num_vertices (int): Number of vertices in the graph.
			edges (np.ndarray): Matrix describing graph connectivity: edges[source, action] = target.
			init_state_dist (np.ndarray): Initial state distribution.
		"""
		# Convert graph specification to MDP
		num_states = num_vertices
		num_actions = edges.shape[1]
		self.transitions = np.zeros((num_states, num_actions, num_states))

		# Build MDP transition matrix from graph adjacency matrix
		for source in range(num_states):
			for action in range(num_actions):
				target = int(edges[source, action])
				self.transitions[source, action, target] = 1

		# Build MDP using super-class's constructor
		super().__init__(self.transitions, num_actions, init_state_dist)


class CommunityGraph(DiGraph):
	"""
		The CommunityGraph class is a particular graph structure governed by small,
		fully-connected neighbourhoods with sparse transitions between them.

		In particular, for each neighbourhood, neighbour ID 0 is the input node, which
		receives transitions from the output nodes of all other neighbourhoods.
		Output nodes are given neighbour ID neighbourhood_size - 1.
	"""

	def __init__(self, num_neighbourhoods, neighbourhood_size, init_state_dist=None):
		"""
		Initialize the CommunityGraph object.

		Args:
			num_neighbourhoods (int): Number of neighbourhoods in the community.
			neighbourhood_size (int): Number of citizens per neighbourhood.
			init_state_dist (np.ndarray): Initial state distribution for the MDP.
		"""
		num_vertices = num_neighbourhoods * neighbourhood_size

		# Transitions to all the neighbours + attempt to hop to other neighbourhood
		num_actions = neighbourhood_size + num_neighbourhoods 

		self.edges = np.zeros((num_vertices, num_actions), dtype=int)
		for nbrhd in range(num_neighbourhoods):
			for nbr in range(neighbourhood_size):
				nbr_vid = self.__class__.nbr_to_vtx(nbr, nbrhd, neighbourhood_size)

				for action in range(num_actions):
					if action <= neighbourhood_size - 1:  # First few actions transition within the neighbourhood
						target_nbr = (nbr + action + 1) % neighbourhood_size
						target_vid = self.__class__.nbr_to_vtx(target_nbr, nbrhd, neighbourhood_size)
						self.edges[nbr_vid, action] = target_vid

					else:  # Remaining actions attempt to transition to another neighbourhood
						if nbr == neighbourhood_size - 1:  # Output node
							target_nbrhd = action - neighbourhood_size
							target_nbr = 0
							target_vid = self.__class__.nbr_to_vtx(target_nbr, target_nbrhd, neighbourhood_size)
							self.edges[nbr_vid, action] = target_vid

						else:  # Non-output node
							self.edges[nbr_vid, action] = nbr_vid

		# Store CG parameters in case someone wants to access them
		self.num_neighbourhoods = num_neighbourhoods
		self.neighbourhood_size = neighbourhood_size

		# Build MDP using super-class's constructor
		super().__init__(num_vertices, self.edges, init_state_dist)

	@staticmethod
	def nbr_to_vtx(nbr, nbrhd, neighbourhood_size):
		"""
		Convert "neighbourhood ID" (i.e., the tuple of neighbourhood ID and neighbour ID) into a graph vertex
		ID.

		Args:
			nbr (int): ID within neighbourhood.
			nbrhd (int): Neighbourhood ID.
			neighbourhood_size (int): Size of all neighbourhoods.

		Returns:
			Graph vertex id (int).

		"""
		return neighbourhood_size * nbrhd + nbr 

	def get_all_transitions(self):
		"""
		Returns all available transitions within the CommunityGraph.

		Returns:
			Ts (list): A list of all valid transitions within the CommunityGraph.
		"""
		Ts = []
		for src_id in range(self.num_states):
			for action in range(self.num_actions):
				transition = (src_id, action, self.edges[src_id, action])
				Ts.append(transition)

		return Ts

	def separate_GR(self, num_iters=None, gamma=None, conv_tol=1e-6, G=None):
		"""
		Return a partial-knowledge version of the true GR for this CommunityGraph, where distances
		are known within but not across communities. The user may provide parameters for the solving
		of the underlying MDP, or they may directly provide a Geodesic representation.

		Args:
			gamma (float): Temporal discount factor used to compute the true Geodesic representation.
			num_iters (int): Amount of iterations to run the value iteration algorithm to compute the true GR.
			conv_tol (float): Early stopping criterion for the value iteration algorithm. It will terminate if
				no state changes by more than conv_tol on successive steps of the algorithm.
			G (np.ndarray): If the user has a Geodesic representation array on hand, they may provide it directly.

		Returns:
			G (np.ndarray): A partial-knowledge Geodesic representation.
		"""
		if G is None:
			G = self.solve_GR(num_iters, gamma, conv_tol)
		else:
			G = G.copy()

		for nbrhd in range(self.num_neighbourhoods):
			min_id = self.nbr_to_vtx(0, nbrhd, self.neighbourhood_size)
			max_id = self.nbr_to_vtx(self.neighbourhood_size - 1, nbrhd, self.neighbourhood_size)

			# Erase everything outside these ids
			G[min_id:max_id + 1, :, :min_id] = 0
			G[min_id:max_id + 1, :, max_id + 1:] = 0

		return G
