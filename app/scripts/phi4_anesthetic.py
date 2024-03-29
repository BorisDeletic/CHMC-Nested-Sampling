import anesthetic as ns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import os

params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
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
}

# Update the default parameters
mpl.rcParams.update(params)


path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/phi4"


def plot_mag(path):
    fname = "Phi4_0.1000_0.0203"

    kappa = 0.2
    lam = 0.0203

    chains = os.path.join(path, fname)
    samples = ns.read_chains(chains)
    posterior = samples.posterior_points()


    ax = sns.histplot(posterior['m'], bins=30, kde=False, color='skyblue')
    # Add a title and labels
    ax.set_title('Posterior Field Distribution\n(Disordered Phase)', fontsize=16)
    ax.set_xlabel(r'$\phi(x)$', fontsize=14)
    ax.set_ylabel('', fontsize=14)

    ax.tick_params(labelleft=False)
    ax.set_yticks([])

    # Customize the grid

    # Remove top and right spines
#    sns.despine()


def save_magnetisations(path):
    data_path = os.path.join(path, 'phase_diagram')
    out_path = os.path.join(path, 'data.csv')
    file_list = os.listdir(data_path)
    files_searched = []

    if (os.path.exists(out_path) == False):
        empty = pd.DataFrame(columns=['kappa', 'lambda', 'mag'])
        empty.to_csv(out_path, index=False)
        print(empty)


    for file in file_list:
        fname = file[:16]

        if fname in files_searched:
            continue
        else:
            files_searched.append(fname)

        mag_data = pd.read_csv(out_path)

        params = file[5:16].split('_')
        params = [float(x) for x in params]

        if (params[0] in mag_data['kappa'].values) and (params[1] in mag_data['lambda'].values):
            continue

        chains = os.path.join(data_path, fname)
        samples = ns.read_chains(chains)
        posterior = samples.posterior_points()

        try:
            mag = abs(posterior['m'])
        except:
            print(params)

        mean_mag = mag.mean()

        data = pd.DataFrame({
            'kappa': params[0],
            'lambda': params[1],
            'mag': mean_mag
        }, index=[0])

        print(mag_data)
        mag_data = pd.concat([mag_data, data], axis=0)
        mag_data.to_csv(out_path, index=False)







def plot_phase_diagram():
    read_file = os.path.join(path, 'data.csv')
    df = pd.read_csv(read_file)

    print(df)

    table = df.pivot('lambda', 'kappa', 'mag')

    #ax = sns.heatmap(table, vmin=0, vmax=15, cmap='mako')
    ax = sns.heatmap(table, vmin=0, vmax=15)

    #ax.locator_params(axis='y', nbins=6)
    #ax.locator_params(axis='x', nbins=5)

    ax.invert_yaxis()
    ax.set_title(r"$\langle M \rangle$" + " Phase Diagram for 32x32 Lattice")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\lambda$")

    ax.set_xticks([0, 8, 16, 24])
    ax.set_xticklabels(['0', '0.1', '0.2', '0.3'])

    ax.set_yticks([0, 13, 27, 39])
    ax.set_yticklabels(['0', '0.01', '0.02', '0.03'])

    # Set the color of the border
    for _, spine in ax.spines.items():
        spine.set_visible(True)


#save_magnetisations(path)

#plot_phase_diagram()
plot_mag(path)

plt.show()



#samples.gui()
#plt.hist(abs(posterior['m']))
#plt.show()
