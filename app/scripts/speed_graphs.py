import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': [6.4, 4.8],
    'figure.dpi': 100,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1,
    'ytick.minor.width': 1,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linewidth': 1,
    'image.cmap': 'viridis',
    'image.interpolation': 'nearest',
    'xtick.bottom': True,
    'ytick.left': True,
    'xtick.labelbottom': True,
    'ytick.labelleft': True
}

mpl.rcParams.update(params)

path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/results/"

polychord_data = pd.read_csv(os.path.join(path, "polychord_phi4_speeds.txt"))
chmc_data = pd.read_csv(os.path.join(path, "dim_bench.txt"))

polychord_data['log_D'] = np.log(polychord_data['D'])
polychord_data['log_tps'] = np.log(polychord_data['tps'])

chmc_data['log_D'] = np.log(chmc_data['D'])
chmc_data['log_tps'] = np.log(chmc_data['tps'])

fig, ax = plt.subplots()

print(polychord_data)

ax.loglog(polychord_data['D'], polychord_data['tps'], linestyle = '', marker='x', label = 'PolyChord', color='tab:blue')
ax.plot(chmc_data['D'], chmc_data['tps'], linestyle='', marker='x', color='tab:orange', label='Constrained HMC')

m,b = np.polyfit(polychord_data['log_D'], polychord_data['log_tps'], 1)
y = np.exp(m * polychord_data['log_D'] + b)
ax.loglog(polychord_data['D'], y, linestyle='--', color='tab:blue', label = 'm = {:.2f}'.format(m))

m,b = np.polyfit(chmc_data['log_D'], chmc_data['log_tps'], 1)
y = np.exp(m * chmc_data['log_D'] + b)
ax.plot(chmc_data['D'], y, linestyle='--', color='tab:orange', label = 'm = {:.2f}'.format(m))


# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_title('Time per sample (ms) vs Dimension')
#
# ax.set_xlabel('Dimension')
# ax.set_ylabel('Time per sample (ms)')

# ax.plot(np.log(polychord_data['D']), np.log(polychord_data['tps']), linestyle = '', marker='x', label = 'PolyChord', color='tab:blue')
# ax.plot(np.log(chmc_data['D']), np.log(chmc_data['tps']), linestyle='', marker='x', color='tab:orange', label='Constrained HMC')
#
# m,b = np.polyfit(np.log(chmc_data['D']), np.log(chmc_data['tps']), 1)
# y = m * np.log(chmc_data['D']) + b
# ax.plot(np.log(chmc_data['D']), y, linestyle='--', color='tab:orange', label = 'm = {:.2f}'.format(m))
#
# m,b = np.polyfit(np.log(polychord_data['D']), np.log(polychord_data['tps']), 1)
# y = m * np.log(polychord_data['D']) + b
# ax.plot(np.log(polychord_data['D']), y, linestyle='--', color='tab:blue', label = 'm = {:.2f}'.format(m))

#ax.set_ylim([0, 1500])
ax.legend(loc='lower right')

ax.set_title('Time per sample (ms) vs Dimension')

ax.set_xlabel('Dimension')
ax.set_ylabel('Time per sample (ms)')

#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.show()