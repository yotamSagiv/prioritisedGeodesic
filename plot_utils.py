import numpy as np
import matplotlib.colorbar as colourbar
import matplotlib.colors as colours
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from copy import copy

from RL_utils import oned_twod


def plot_replay(gridworld, replay_sequence, ax=None, figsize=(12, 12), highlight_last=False, draw_gw=True):
	"""
		Given a GridWorld object, and a list of replayed states in that GridWorld, plot the sequence in a nice
		and pretty way.

		Params:
			gridworld: GridWorld object
			replay_seequence: N x 3 array, where each row is (start state, action, successor state)
	"""

	# Paint the grid world to the figure
	if ax is None or draw_gw:
		ax = gridworld.draw(use_reachability=True, ax=ax, figsize=figsize)

	ax.set_aspect('equal')

	## Now add arrows for replayed states
	# Colours!
	arrow_colours = plt.cm.winter(np.linspace(0, 1, replay_sequence.shape[0]))
	CENTRE_OFFSET = 0.5  # oned_twod gives the coordinate of the top left corner of the state
	for i in range(replay_sequence.shape[0]):
		# Get plotting coordinates
		start, action, successor = replay_sequence[i, :]
		start_y, start_x = np.array(oned_twod(start, gridworld.width, gridworld.height)) + CENTRE_OFFSET
		succ_y, succ_x = np.array(oned_twod(successor, gridworld.width, gridworld.height)) + CENTRE_OFFSET

		# Plot
		if i == replay_sequence.shape[0] - 1 and highlight_last:
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y,
					 length_includes_head=True, head_width=0.25, color='r')
		else:
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y,
					 length_includes_head=True, head_width=0.25, color=arrow_colours[i])

	# Add colour bar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cbar = colourbar.ColorbarBase(cax, cmap=plt.cm.winter, orientation='vertical', ticks=[0, 1])
	cbar.ax.set_yticklabels(['start', 'end'])

	return ax


def plot_traj(gridworld, state_sequence, ax=None, figsize=(12, 12)):
	"""
		Given a GridWorld object, and a list of replayed states in that GridWorld, plot the sequence in a nice
		and pretty way.

		Params:
			gridworld: GridWorld object
			state_sequence: N x 1 array, consisting of visited state sequence
	"""
	# Paint the grid world to the figure
	ax = gridworld.draw(use_reachability=True, ax=ax, figsize=figsize)
	
	## Now add arrows for traversed states
	# Colours!
	arrow_colours = plt.cm.winter(np.linspace(0, 1, len(state_sequence)))
	CENTRE_OFFSET = 0.5  # oned_twod gives the coordinate of the top left corner of the state
	for i in range(len(state_sequence) - 1):
		# Get plotting coordinates
		start = state_sequence[i]
		successor = state_sequence[i + 1]
		start_y, start_x = np.array(oned_twod(start, gridworld.width, gridworld.height)) + CENTRE_OFFSET
		succ_y, succ_x = np.array(oned_twod(successor, gridworld.width, gridworld.height)) + CENTRE_OFFSET

		# Plot
		ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color=arrow_colours[i])

	# Add colour bar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cbar = colourbar.ColorbarBase(cax, cmap=plt.cm.winter, orientation='vertical', ticks=[0, 1])
	cbar.ax.set_yticklabels(['start', 'end'])

	return ax


def plot_need_gain(gridworld, transitions, need, gain, MEVB, specials=None, params=None, fig=None, axes=None, use_cbar=False,
				   custom_need_cmap=None):
	# Dumb input validation
	if params is None:
		params = {}

	if fig is None or axes is None:
		fig, axes = plt.subplots(1, 3, figsize=(18, 6))

	## Need
	# Plot need by shading in states
	ax = axes[0]
	ax.set_aspect('equal')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.axis('off')
	ax.set_title('Need')
	gridworld.draw(use_reachability=True, ax=ax)

	# Grab boundaries
	min_need = np.min(need[np.nonzero(need)])  # Dumb hack so that need = 0 states don't appear slightly red
	max_need = np.max(need)
	alpha_fac = 1
	distinguish_max = False
	distinguish_chosen = False

	if 'min_need' in params.keys():
		min_need = params['min_need']
	if 'max_need' in params.keys():
		max_need = params['max_need']
	if 'alpha_fac' in params.keys():
		alpha_fac = params['alpha_fac']
	if 'distinguish_max' in params.keys():
		distinguish_max = params['distinguish_max']
	if 'distinguish_chosen' in params.keys():
		distinguish_chosen = params['distinguish_chosen']

	norm_need = colours.Normalize(vmin=min_need, vmax=max_need)(need)

	# Build custom palette without dumb red bottom boundary
	if custom_need_cmap:
		palette = custom_need_cmap
	else:
		palette = copy(plt.get_cmap('Reds'))
		
	palette.set_under('white', 1.0)

	# Get colours for each state
	state_colours = palette(norm_need).reshape(-1, 4)
	for state in range(gridworld.num_states):
		if hasattr(gridworld, 'banned_states') and state in gridworld.banned_states:
			continue
		row, col = oned_twod(state, gridworld.width, gridworld.height)
		rect = patches.Rectangle((col, row), 1, 1, facecolor=state_colours[state])
		ax.add_patch(rect)

	# Add colour bar
	if use_cbar:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		cbar = colourbar.ColorbarBase(cax, cmap=palette, orientation='vertical', ticks=[0, 1])
		cbar.ax.set_yticklabels(['%.3f' % min_need, '%.3f' % max_need])

	## Gain
	# Plot gain by shading arrows
	ax = axes[1]
	ax.set_aspect('equal')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.axis('off')

	ax.set_title('Gain')
	gridworld.draw(use_reachability=True, ax=ax)

	# Grab boundaries
	min_gain = np.min(gain) 
	max_gain = np.max(gain)

	if 'min_gain' in params.keys():
		min_gain = params['min_gain']
	if 'max_gain' in params.keys():
		max_gain = params['max_gain']

	norm_gain = colours.Normalize(vmin=min_gain, vmax=max_gain)(gain)
	gain_colours = plt.cm.winter(norm_gain).reshape(-1, 4)
	gain_colours[:, 3] = norm_gain / alpha_fac  # Modulate alpha in accordance with gain as well
	CENTRE_OFFSET = 0.5  # oned_twod gives the coordinate of the top left corner of the state
	for tdx, transition in enumerate(transitions):
		s_k, a_k, s_kp = transition
		start_y, start_x = np.array(oned_twod(s_k, gridworld.width, gridworld.height)) + CENTRE_OFFSET
		succ_y, succ_x = np.array(oned_twod(s_kp, gridworld.width, gridworld.height)) + CENTRE_OFFSET

		# Plot
		if abs(np.max(norm_gain) - norm_gain[tdx]) < 1e-8 and distinguish_max:  # Distinguish the maximal gain transitions
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color='r')
		else:
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color=gain_colours[tdx])

	## MEVB
	# Plot MEVB by shading arrows
	ax = axes[2]
	ax.set_aspect('equal')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.axis('off')

	ax.set_title('EVB')
	gridworld.draw(use_reachability=True, ax=ax)

	# Grab boundaries
	min_MEVB = np.min(MEVB) 
	max_MEVB = np.max(MEVB)

	if 'min_MEVB' in params.keys():
		min_MEVB = params['min_MEVB']
	if 'max_MEVB' in params.keys():
		max_MEVB = params['max_MEVB']

	norm_MEVB = colours.Normalize(vmin=min_MEVB, vmax=max_MEVB)(MEVB)
	MEVB_colours = plt.cm.winter(norm_MEVB).reshape(-1, 4)
	MEVB_colours[:, 3] = norm_MEVB / alpha_fac
	CENTRE_OFFSET = 0.5  # oned_twod gives the coordinate of the top left corner of the state
	for tdx, transition in enumerate(transitions):
		s_k, a_k, s_kp = transition
		start_y, start_x = np.array(oned_twod(s_k, gridworld.width, gridworld.height)) + CENTRE_OFFSET
		succ_y, succ_x = np.array(oned_twod(s_kp, gridworld.width, gridworld.height)) + CENTRE_OFFSET

		# Plot
		if specials is not None and transition in specials:  # Custom distinction for a set of transitions
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color='k')
		elif abs(np.max(norm_MEVB) - norm_MEVB[tdx]) < 1e-8 and distinguish_chosen:  # Distinguish the eventually-chosen transition
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
					  length_includes_head=True, head_width=0.25, color='r')
		else:
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
					  length_includes_head=True, head_width=0.25, color=MEVB_colours[tdx])


def plot_state_metric(gridworld, metric, save=True, savename='', use_cbar=False, wall_width=2, cmap='Reds', under='white',
					  ax=None, custom_cmap=None):
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(12, 12))

	ax.set_aspect('equal')
	ax.set_axis_off()
	ax, unreachable_states = gridworld.draw(use_reachability=True, ax=ax, wall_width=wall_width, return_unreachable=True)

	# Grab boundaries
	min_metric = np.min(metric[np.nonzero(metric)])  # Dumb hack so that metric = 0 states don't appear slightly red
	max_metric = np.max(metric)

	metric_cols = colours.Normalize(vmin=min_metric, vmax=max_metric)(metric)

	# Build custom palette without dumb red bottom boundary
	if custom_cmap:
		palette = custom_cmap
	else:
		palette = copy(plt.get_cmap(cmap))

	palette.set_under(under, 1.0)

	# Get colours for each state
	state_colours = palette(metric_cols).reshape(-1, 4)
	for state in range(gridworld.num_states):
		if hasattr(gridworld, 'banned_states') and state in gridworld.banned_states:
			continue
		if state in unreachable_states:
			continue

		row, col = oned_twod(state, gridworld.width, gridworld.height)
		rect = patches.Rectangle((col, row), 1, 1, facecolor=state_colours[state])
		ax.add_patch(rect)

	# Add colour bar if desired
	if use_cbar:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		cbar = colourbar.ColorbarBase(cax, cmap=palette, orientation='vertical', ticks=[0, 1])
		cbar.ax.set_yticklabels(['%.3f' % min_metric, '%.3f' % max_metric])

	# Save if desired
	if save:
		plt.savefig(savename, transparent=True, bbox_inches=0, pad_inches=0)


if __name__ == '__main__':
	pass
