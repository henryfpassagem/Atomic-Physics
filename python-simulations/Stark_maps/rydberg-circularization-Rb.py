#! /usr/bin/python3.8

import os
import pint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from lib.Plotting import ThesisPlot, draw_double_arrow, draw_text
from lib.StarkMap import ModifiedStarkMap
from lib.ureg import ureg
from arc import Rubidium85
import json

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from brokenaxes import brokenaxes


LOAD_FROM_FILE = True
cached_data_file = os.path.join('..', 'data', 'cached-plot-data', 'PlotRbCircularization.json')

def plot_state(ax, state: tuple, color: str, state_labels: list, state_indices: list, energies: list):
    """Function for plotting an energy level. state_labels, state_indices and energies come from ModifiedStarkMap. """

    # Set length of line
    dash_length = 0.75
    dash = np.linspace(-dash_length / 2, dash_length / 2, 10)
    # Get m
    m = state[-1]
    # Get the index of the state
    index = state_labels[m].index(state)
    # Get the energy of the state
    E_state = (energies[m][state_indices[m][index]] / ureg('c')).to('1/cm')
    # Plot the state
    ax.plot(m + dash, E_state.magnitude * np.ones(dash.shape), color=color, lw=2)

# Set the plot
width = 15 # in cm

tp = ThesisPlot()
tp.set_figsize(width=width, ratio=3/2)
tp.set_x_major_ticks(direction='out', top=False)
tp.set_x_minor_ticks(visible=False)
tp.set_y_major_ticks(direction='out', right=False)
tp.set_borders(bottom=0.15, left=0.15)

###### 1. Calculate the energy for all m states of the n = 51 manifold at 1 V/cm ######
E = 1 * ureg('V/cm')
ms = np.arange(0, 51, 1) # m = 0, ..., 50

if LOAD_FROM_FILE:
    # Read data
    with open(cached_data_file, 'r') as f:
        data = json.load(f)
    # Turn it into correct format
    # Add units to energies
    energies = [np.array(x) * ureg('GHz') for x in data['energies']]
    # turn state labels into tuples
    state_labels = data['state_labels']
    for m in range(len(state_labels)):
        for i in range(len(state_labels[m])):
            state_labels[m][i] = tuple(state_labels[m][i])
    state_indices = data['state_indices']
else:
    # Allocate memory
    energies = []
    state_labels = []
    state_indices = []

    # Iterate over all m
    for m in ms:
        # Initialize Stark map
        stark_map = ModifiedStarkMap(Rubidium85(), 45, 55, 54, m)
        # Find energies for field E
        e, _ = stark_map.diagonalize(E)
        energies.append(e)
        # Find labels of the states
        labels, indices = stark_map.find_state_labels(51, m, debug=False)
        # Save them
        state_labels.append(labels)
        state_indices.append(indices)    

    # Convert all np.int64 to int
    for m in range(len(state_labels)):
        for i in range(len(state_labels[m])):
            state_labels[m][i] = tuple(map(int, state_labels[m][i]))
    state_indices = [list(map(int, x)) for x in state_indices]

    # Write data to file
    data = {
        'energies': [x.to('GHz').magnitude.tolist() for x in energies],  
        'state_labels': state_labels, 
        'state_indices': state_indices
    }
    print([type(x) for x in state_labels[0][0]])
    print(type(state_indices[0][0]))

    # Save everything
    with open(cached_data_file, 'w') as f:
        json.dump(data, f)

###### 2. Select the states on the diagonals with constant value of n1 ######
# All states on the diagonal ending at (51, 0, 0, 50)
ladder_states = [(51, 0, 51 - m - 1, m) for m in np.arange(50, 4, -1)] + \
                [(51, 4, 4)] + \
                [(51, 3, 3), (51, 3, 2)]
n1_0_states = [(51, 3, 1), (51, 3, 0)]

# All states on the diagonal ending at (51, 1, 0, 49)
n1_1_ladder = [(51, 1, 50 - m - 1, m) for m in np.arange(49, 3, -1)] + \
              [(51, 4, m) for m in np.arange(3, -1, -1)]

# All states on the diagonal ending at (51, 2, 0, 48)
n1_2_ladder = [(51, 2, 49 - m - 1, m) for m in np.arange(48, 2, -1)] + \
              [(51, 3, 45, 2), (51, 4, 45, 1), (51, 5, 45, 0)]

# All states on the diagonal ending at (51, 3, 0, 47)
n1_3_ladder = [(51, 3, 48 - m - 1, m) for m in np.arange(47, 1, -1)] + \
              [(51, 4, 44, 2), (51, 5, 44, 1), (51, 6, 44, 0)]

# Choose the states we plot
# The highlighted ones
highlighted_states = ladder_states[:4] + ladder_states[-4:]
# The rest
blue_states = n1_0_states + \
              n1_1_ladder[:3] + n1_1_ladder[-6:] + \
              n1_2_ladder[:2] + n1_2_ladder[-4:] + \
              n1_3_ladder[:1] + n1_3_ladder[-2:]

###### 3. Make broken axis plot ######
bax = brokenaxes(
    xlims=((-0.5, 5.5), (46.5, 50.5)), 
    ylims=((-42.36, -42.329), (-42.21, -42.18)), 
    hspace=.1, wspace=0.8,
    d=.008
)

for state in highlighted_states:
    plot_state(bax, state, 'C1', state_labels, state_indices, energies)

for state in blue_states:
    plot_state(bax, state, 'C6', state_labels, state_indices, energies)


# x axis: Use global label because labelpad cannot have different values for x and y
#bax.set_xlabel('|m|', labelpad=10)
plt.annotate('$|m|$', (0.536, 0.05), xycoords='figure fraction')

# y axis
bax.set_ylabel('Energy (cm$^{-1}$)', labelpad=40)

###### 4. Label states ######
ax = bax.axs[1]
EC = (energies[50][state_indices[50][0]] / ureg('c')).to('1/cm').magnitude
ax.annotate('$\\mid 51C \\, \\rangle$', (50 - 0.55, EC + 0.001))

ax = bax.axs[2]
Ef = (energies[2][state_indices[2][-2]] / ureg('c')).to('1/cm').magnitude
ax.annotate('$\\mid 51f \\, \\rangle$', (2 - 0.5, Ef + 0.001))

###### 5. Make dots between low m and high m states ######
plt.annotate('...', (0.575, 0.54), rotation=31, xycoords='figure fraction', fontsize=18, color='C1')
plt.annotate('...', (0.575, 0.62), rotation=31, xycoords='figure fraction', fontsize=18, color='C6')
plt.annotate('...', (0.575, 0.70), rotation=31, xycoords='figure fraction', fontsize=18, color='C6')
plt.annotate('...', (0.575, 0.78), rotation=31, xycoords='figure fraction', fontsize=18, color='C6')

###### 6. Make inset ######
axins = inset_axes(bax.axs[0], width=1.3, height=1.)

# Plot highlighted states of the ladder
for state in ladder_states[-3:]:
    plot_state(axins, state, 'C1', state_labels, state_indices, energies)

# Plot the other states
for state in n1_0_states[:1] + n1_1_ladder[-3:-1]:
    plot_state(axins, state, 'C6', state_labels, state_indices, energies)

###### 7. Plot arrows between states ######
# Define the arrows we want to plot between the ladder states
arrows = [
    ((51, 3, 1), (51, 3, 2)),
    ((51, 3, 2), (51, 3, 3)),
    ((51, 3, 3), (51, 4, 4))
]

# Plot the arrows
for a, b in arrows:
    # Get the m's
    m_a = a[-1]
    m_b = b[-1]
    # Get the indices
    index_a = state_labels[m_a].index(a)
    index_b = state_labels[m_b].index(b)
    # Get the energies
    E_a = (energies[m_a][state_indices[m_a][index_a]] / ureg('c')).to('1/cm')
    E_b = (energies[m_b][state_indices[m_b][index_b]] / ureg('c')).to('1/cm')
    # Calculate transition frequency
    nu = (np.abs(E_a - E_b) * ureg('c')).to('MHz')
    # Set endpoints of arrow
    xy_a = (m_a, E_a.magnitude)
    xy_b = (m_b, E_b.magnitude)
    
    # Draw the arrow
    draw_double_arrow(xy_a, xy_b, linewidth=0.5, head_width=3, head_length=3, cut=0.1)
    # Add frequency of the transition
    xy_freq = (xy_a[0] + 1.3, xy_a[1] + 0.8 * (xy_b[1] - xy_a[1]) / 2)
    draw_text(f'{nu.magnitude:.0f} MHz', xy_freq, fs=8, zorder=-1)

# Set x axis of inset
axins.set_xlim([0.25, 5.1])
# Set y axis of inset
axins.set_ylim([-42.351, -42.339])

# Save the plot
tp.save_figure()
