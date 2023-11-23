import anesthetic as ns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

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


def exp(x, a, m, c):
    return a * np.exp(-m * x) + c

#path = "/rds/user/bd418/hpc-work/correlation"
path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/phi4/correlation"
path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/phi4/correlation_exact"

R = 128

file_list = sorted(os.listdir(path))
files_searched = []

print(file_list)

def correlationLength(correlations):

    log_correlations = np.log(np.abs(correlations[:len(correlations)//2]))

    print(log_correlations)

    xis = {}
    kappas = []
    fig, ax = plt.subplots()
    for kappa in log_correlations.columns.values:
        m,b = np.polyfit(log_correlations.index.values, log_correlations[kappa], 1)

        xi = -1 / m

        xis[kappa] = xi
        kappas.append(kappa)
        ax.plot(log_correlations[kappa], label="k={:.5f}, xi={:.3f}".format(float(kappa), xi))
    #    ax.plot(kappa, , label="k={:.5f}, xi={:.3f}".format(kappa, xi))

    df = pd.DataFrame(xis.items(), columns=['kappa', 'xi'])
    return df

   # ax.figure.savefig('log_corr.png')

def read_correlation_data():
    correlation_samples = pd.DataFrame()
   # mags = {}

    for file in file_list:
        fname = file[:22]

        if fname in files_searched or fname == '.DS_Store':
            continue
        else:
            files_searched.append(fname)

        kappa = float(fname.split("_")[1])
        l = fname.split("_")[2]

        chains = os.path.join(path, fname)
        samples = ns.read_chains(chains)
        posterior = samples.posterior_points()

        mean_mag = abs(posterior['mag']).mean()

        print("kappa = {}, meanmag = {}".format(kappa, mean_mag))
#        correlations = [posterior["c_0"].mean()]
        correlations = []
        for r in range(1, R):
            c_key = "c_{}".format(r)

            mean_correlation = posterior[c_key].mean() - mean_mag*mean_mag
           # print(c_key)
            #print(mean_correlation)
            correlations.append(mean_correlation)

        correlations /= posterior["c_0"].mean()
        correlation_samples[kappa] = correlations
        #mags[kappa] = mean_mag

    correlation_samples.to_csv("correlation_data.csv", index=False)
   # mag_df = pd.DataFrame(mags)
   # mag_df.to_csv("mag_data.csv", index=False)


def plot_xi(xis):
    fig, ax = plt.subplots()
    print(xis)
    xis['kappa'] = pd.to_numeric(xis['kappa'])
    xis = xis[(xis['kappa'] > 0.11747) & (xis['kappa'] < 0.1176)]

    linearxi = xis[(xis['kappa'] > 0.11753)]
    m,b = np.polyfit(linearxi['kappa'], linearxi['xi'], 1)

    y = m * linearxi['kappa'] + b

    ax.plot(xis['kappa'], xis['xi'], marker='x', linestyle = '')
    ax.axvline(0.117534, linestyle = '--', color='black')
   # ax.plot(linearxi['kappa'], y, linestyle = '--', color='black')

    ax.set_xlabel(r'$\kappa$')
    ax.set_ylabel(r'$\xi$')

    ax.set_title(r'Correlation Length vs $\kappa$' + '\n128x128 Lattice')


def plot_correlation(correlation_samples):
    fig, ax = plt.subplots()

    kappa1 = '0.1175'
    kappa2 = '0.11754'
    init_period = R//2
    init_decay1 = correlation_samples[kappa1][:init_period]
    init_decay2 = correlation_samples[kappa2][:init_period]

    x = np.arange(0, init_period, 1)
    popt, pcov = curve_fit(exp, init_decay1.index.values, init_decay1.values)
    a, m, c = popt[0], popt[1], popt[2]
    y1 = exp(x, a, m, c)
    popt, pcov = curve_fit(exp, init_decay2.index.values, init_decay2.values)
    a, m, c = popt[0], popt[1], popt[2]
    y2 = exp(x, a, m, c)

    ax.plot(x, y1, linestyle='--', color='tab:blue')
    ax.plot(x, y2, linestyle='--', color='tab:red')

    ax.plot(init_decay1, label=r"$\kappa$={:.5f}".format(float(kappa1)), linestyle='', marker='x', color='tab:blue')
    ax.plot(init_decay2, label=r"$\kappa$={:.5f}".format(float(kappa2)), linestyle='', marker='x', color='tab:red')

    ax.set_title('Correlation functions\n128x128 Lattice')
    ax.set_xlabel('r')
    ax.set_ylabel('C(r)')
    ax.legend(loc="upper right")


def plot_all_correlation(correlation_samples):
    fig, ax = plt.subplots()

    init_period = R//2
    for kappa in correlation_samples.columns.values:
        init_decay = correlation_samples[kappa][:init_period]

        x = np.arange(0, init_period, 1)
        popt, pcov = curve_fit(exp, init_decay.index.values, init_decay.values)
        a, m, c = popt[0], popt[1], popt[2]
        y1 = exp(x, a, m, c)

        ax.plot(x, y1, linestyle='--', color='tab:blue')

        ax.plot(init_decay, label="k={:.6f}".format(float(kappa)), linestyle='', marker='.', color='tab:blue')

    ax.set_title('Correlation functions\n128x128 Lattice')
    ax.set_xlabel('r')
    ax.set_ylabel('C(r)')
    ax.legend(loc="upper right")


def plot_512_correlation(correlation_samples, kappas):
    fig, ax = plt.subplots()

    color = iter(['tab:blue', 'tab:red'])
    for kappa in kappas:
        init_period = R//2
        init_decay = correlation_samples[kappa][:init_period]

        x = np.arange(0, init_period, 1)
        popt, pcov = curve_fit(exp, init_decay.index.values, init_decay.values)
        a, m, c = popt[0], popt[1], popt[2]
        y1 = exp(x, a, m, c)

        c = next(color)
        ax.plot(x, y1, linestyle='--', color=c)

        ax.plot(init_decay, label=r"$\kappa$={:.6f}".format(float(kappa)), linestyle='', marker='x', markersize = 2, color=c)

    ax.set_title('Correlation functions\n512x512 Lattice')
    ax.set_xlabel('r')
    ax.set_ylabel('C(r)')
    ax.legend(loc="upper right")


read_correlation_data()

correlation_samples = pd.read_csv("correlation_data.csv")

plot_correlation(correlation_samples)
#plot_all_correlation(correlation_samples)

kappas = ['0.1175', '0.117502']
#plot_512_correlation(correlation_samples, kappas)

#xis = correlationLength(correlation_samples)
#plot_xi(xis)
# kappa = '0.1174'
# kappa = '0.1175'
# kappa = '0.1173'
# kappa = '0.11754'


plt.show()
#ax.figure.savefig('correlations.png')



