import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy

# can get this number from .stats
# nlive = 500
n=32

root = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/"
phase_folder = "phase_diagram/"
scaling_folder = "scaling/"
file = "Phi4_posterior_sampling"

def get_stats(fname):
    f = open(fname + ".stats")
    nlive_str = f.readline()
    iters_str = f.readline()
    f.close()

    num_live = int(nlive_str.split(" ")[-1])
    iters = int(iters_str.split(" ")[-1])

    return num_live, iters

def read_file(fname, params):
    nlive, iters = get_stats(fname)
    # print(fname)
    df = pd.read_csv(fname + ".posterior", names=['log_weight', "log_like", "mag", "mag_squared"], header=None, sep=" ", index_col=False)
    # Z = np.exp(scipy.special.logsumexp(df['log_weight']))
    # print(Z)
    df['t'] = np.log(nlive/(nlive+1))
    df['logX'] = df['t'].cumsum()
    logXp = df['logX'].shift(1, fill_value=0)
    logXm = df['logX'].shift(-1, fill_value=-np.inf)
    df['logdX'] = np.log(1 - np.exp(logXm-logXp)) + logXp - np.log(2)
    df.drop(df.tail(1).index,inplace=True)

    df['logW'] = df['logdX'] + df['log_like']

    logZ = scipy.special.logsumexp(df['logW'])

    df['logW'] -= logZ

    # df['weight'] = np.exp(df['log_weight']) / Z

    # df['log_weighted_mag'] = df['logW'] + np.log(np.abs(df['mag']))
    df['weighted_abs_phi'] = np.exp(df['logW']) * np.abs(df['mag'])
    df['weighted_phi'] = np.exp(df['logW']) * df['mag']
    df['weighted_phi_squared'] = np.exp(df['logW']) * df['mag']**2
    df['weighted_phi_4'] = np.exp(df['logW']) * df['mag']**4
    # print(df['log_weighted_mag'])

    # mag = np.exp(scipy.special.logsumexp(df['log_weighted_mag']))
    abs_phi = np.sum(df['weighted_abs_phi'])
    phi = np.sum(df['weighted_phi'])
    phi_squared = np.sum(df['weighted_phi_squared'])
    phi_4 = np.sum(df['weighted_phi_4'])

    chi = n**2 * (phi_squared - abs_phi**2)

    # binder = 1 - phi_4 / (3 * phi_squared**2)
    # print(chi)
    results = {
        'kappa': params[0],
        'lambda': params[1],
        'mean_phi': phi,
        'mean_mod_phi': abs_phi,
        'phi_squared': phi_squared,
        'chi': chi,
        # 'U': binder
    }

    return results


def load_phase_data():
    data = []
    files_searched = []

    file_list = os.listdir(root + phase_folder)
    for file in file_list:
        fname = file[:20]

        params = file[5:20].split('_')
        params = [float(x) for x in params]

        try:
            out = read_file(root + phase_folder + fname, params)
            data.append(out)
        except:
            continue

        files_searched.append(fname)

    phase_data = pd.DataFrame(data).sort_values(by='kappa')

    return phase_data


def scaling_ansatz(x, a, b, c):
    return a + b * x**c

def residuals(c, x, y):
    yfit = scaling_ansatz(x, c[0], c[1], c[2])
    rs = (y - yfit)**2
    # kappa_crit = y - c[0]
    #
    # rs = np.log(kappa_crit) - (c[1] - c[2] * np.log(x))

    return rs.sum()

def load_scaling_data():
    scaling_dfs = {}

    subfolder_list = os.listdir(root + scaling_folder)
    for n in subfolder_list:
        files_searched = []
        data = []
        file_list = os.listdir(root + scaling_folder + n)

        for file in file_list:
            fname = file[:22]

            if fname in files_searched:
                continue

            # if (fname == "Phi4_0.200283_0.100000"):
            #     print("Phi4_0.200283_0.100000", n)

            path = root + scaling_folder + n + "/" + fname

            params = file[5:22].split('_')
            params = [float(x) for x in params]

            out = read_file(path, params)
            data.append(out)

            files_searched.append(fname)

        scaling_data = pd.DataFrame(data).sort_values(by='kappa')
        print(n)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(scaling_data.loc[scaling_data['kappa'] == 0.200283])
        print()

        scaling_dfs[int(n)] = scaling_data

    return scaling_dfs



def find_critical_point(data, n):
    # coeff = np.polyfit(data['kappa'], data['chi'], 50)
    # spline = np.poly1d(coeff)
    # print(data[['kappa', 'chi']])

    f = scipy.interpolate.splrep(data['kappa'], data['chi'], s=1)

    kappa = np.linspace(data['kappa'].min(), data['kappa'].max(), 1000)
    chi_fit = scipy.interpolate.BSpline(*f)(kappa)

    idx = np.argmax(chi_fit)
    kappa_c = kappa[idx]
    chi_c = chi_fit[idx]

    fig, ax = plt.subplots()
    ax.plot(data['kappa'], data['chi'], linestyle='', marker='x')
    # ax.plot(kappa, chi_fit, linestyle='--')
    ax.set_title('Chi Vs Kappa, n = {}'.format(n))
    # ax.plot(kappa, chi_ansatz(kappa, popt[0], popt[1], popt[2], popt[3], popt[4]), linestyle='--')

    return kappa_c, chi_c

def plot_kappa_scaling(critical):
    fig, ax = plt.subplots()
    ax.plot(critical['inverse_n'], critical['kappa'], linestyle='', marker='x')

    # res = scipy.optimize.minimize(residuals, [1,1,-1],
    #                                      args=(critical['n'], critical['kappa']))
    popt, pcov = scipy.optimize.curve_fit(scaling_ansatz, critical['inverse_n'], critical['kappa'], maxfev=5000)
    print(popt)
    x = np.linspace(0, critical['inverse_n'].max() * 1.1)
    y = scaling_ansatz(x, popt[0], popt[1], popt[2])

    ax.plot(x, y, linestyle = '--')
    ax.axvline(0, linestyle='--', color='black')

    ax.legend()
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_title('Critical Kappa vs 1/L (inverse lattice size)')
    ax.set_xlabel('1 / L (lattice size)')
    ax.set_ylabel('Kappa critical')


def plot_chi_scaling(critical):
    fig, ax = plt.subplots()
    ax.plot(critical['inverse_n'], critical['chi'], linestyle='', marker='x')
    # ax.set_title('Susceptibility vs Kappa, n = {}'.format(n))
    # ax.set_xlabel('Kappa')
    # ax.set_xlabel('Susceptibility (Chi)')

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_title('Critical Chi vs 1 / L (inverse lattice size)')
    ax.set_xlabel('L (inverse lattice size)')
    ax.set_ylabel('Chi critical')



def scaling_analysis():
    scaling_data = load_scaling_data()
    # scaling_data.to_csv('scaling_data.csv', index=False)
    # scaling_data = pd.read_csv('scaling_data.csv')

    sizes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    critical = pd.DataFrame(columns=['kappa', 'chi', 'n', 'inverse_n'])

    for n in sizes:
        data = scaling_data[n]

        kappa_c, chi_c = find_critical_point(data, n)
        critical.loc[len(critical)] = [kappa_c, chi_c, n, 1/n]

    plot_kappa_scaling(critical)
    plot_chi_scaling(critical)



def phase_diagram(load = False):
    if load:
        phase_data = load_phase_data()
        phase_data.to_csv('phase_data.csv', index=False)
    phase_data = pd.read_csv('phase_data.csv')

    # table = phase_data.pivot('lambda', 'kappa', 'mean_mod_phi')
    table = phase_data.pivot_table(index='lambda', columns='kappa', values='mean_mod_phi')
    # interp_df = table.interpolate(method='linear', axis=1)

    print(table)

    ax = sns.heatmap(table, vmin=0, vmax=3, cmap='mako')
    # ax = sns.heatmap(interp_df, vmin=0, vmax=3)

    #ax.locator_params(axis='y', nbins=6)
    #ax.locator_params(axis='x', nbins=5)

    ax.invert_yaxis()
    ax.set_title(r"$\langle M \rangle$" + " Phase Diagram for 32x32 Lattice")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\lambda$")




# print(phase_data)

# fig, ax = plt.subplots()
# ax.plot(phase_data['kappa'], phase_data['chi'], linestyle='', marker='x')
# #
# fig, ax = plt.subplots()
# ax.plot(phase_data['kappa'], phase_data['U'], linestyle='', marker='x')
#
# fig, ax = plt.subplots()
# ax.plot(phase_data['kappa'], phase_data['mean_mod_phi'], linestyle='', marker='x')
#
# fig, ax = plt.subplots()
# ax.hist(df['mag'], weights=df['weight'], bins=100)

scaling_analysis()
# phase_diagram()
# print(read_file(root + file, [0.35, 0.1]))
# print(read_file(root + "scaling/80/Phi4_0.200283_0.100000", [0.35, 0.1]))
#chi 1447
plt.show()
