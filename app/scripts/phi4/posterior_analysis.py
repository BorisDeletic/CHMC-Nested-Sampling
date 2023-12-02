import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy

# can get this number from .stats
# nlive = 500
# n=32

root = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/"
# root="/home/bd418/rds/hpc-work/"
phase_folder = "phase_diagram/"
scaling_folder = "scaling/"
file = "Phi4_posterior_sampling"


def load_phase_data():
    n = 32
    data = []
    files_searched = []

    file_list = os.listdir(root + phase_folder)
    for file in file_list:
        fname = file[:20]

        params = file[5:20].split('_')
        params = [float(x) for x in params]
        params.append(n)

        try:
            out = read_file(root + phase_folder + fname, params)
            data.append(out)
        except:
            continue

        files_searched.append(fname)

    phase_data = pd.DataFrame(data).sort_values(by='kappa')

    return phase_data

def posterior_points(data, logW):
    w = np.exp(logW)
    u = np.random.rand(len(w))

    neff = 1 / np.max(w)

    W = w * neff / w.sum()

    fraction, integer = np.modf(W)
    extra = (u < fraction).astype(int)
    equal_weights = (integer + extra).astype(int)

    equal_data = np.repeat(data, equal_weights).reset_index(drop=True)

    # fig, ax = plt.subplots()
    # ax.plot(equal_data.reset_index(drop=True), linestyle='', marker='x')

    return equal_data


def get_stats(fname):
    f = open(fname + ".stats")
    nlive_str = f.readline()
    iters_str = f.readline()
    f.close()

    num_live = int(nlive_str.split(" ")[-1])
    iters = int(iters_str.split(" ")[-1])

    return num_live, iters


def autocorrelation_weighted(observable, logW):
    corrs = []
    for shift in range(1, 1000):
        time_delayed_product = (observable[shift:] * observable[:-shift]).dropna()
        time_delayed_logW = (logW[shift:] + logW[:-shift]).dropna()
        # time_delayed_product = (observable * observable).dropna()
        # time_delayed_logW = (logW + logW).dropna()

        observable_sqd = np.average(observable, weights = np.exp(logW))**2
        correlation = np.average(time_delayed_product, weights=np.exp(time_delayed_logW)) - observable_sqd

        corrs.append(correlation)

    return corrs


def autocorrelation(observable):
    corrs = []
    for shift in range(1, 1000):
        time_delayed_product = (observable[shift:] * observable[:-shift]).dropna()

        observable_sqd = np.mean(observable) ** 2
        correlation = np.mean(time_delayed_product) - observable_sqd

        corrs.append(correlation)
    # shift = 1
    # time_delayed_product = (observable[shift:] * observable[:-shift]).dropna()
    # print(time_delayed_product)

    return corrs


# def correlation_function(field, logW):
#     corrs = []
#     for r in range(1, n//2):
#         hor_shifted_product = observable * np.roll(observable, r, 0)
#         # time_delayed_logW = (logW[shift:] + logW[:-shift]).dropna()
#         # time_delayed_product = (observable * observable).dropna()
#         # time_delayed_logW = (logW + logW).dropna()
#
#         observable_sqd = np.average(observable, weights = np.exp(logW))**2
#         correlation = np.average(hor_shifted_product, weights=np.exp(logW)) - observable_sqd
#
#         corrs.append(correlation)
#
#     return corrs

def plot_observables(df):

    # mag = np.average(np.abs(df['mag']), weights=np.exp(df['logW']))
    # mag_squared = np.average(df['mag']**2, weights=np.exp(df['logW']))

    fig, ax = plt.subplots()
    # ax.hist(np.abs(df['mag']), bins=30, weights=np.exp(df['logW']))
    ax.hist(df['mag'], bins=100, weights=np.exp(df['logW']))
    # ax.plot(np.abs(equal_mags))
    # ax.plot(np.exp(df['logW']))


def read_file(fname, params):
    nlive, iters = get_stats(fname)
    # print(fname)
    df = pd.read_csv(fname + ".posterior", names=['log_weight', "log_like", "mag", "mag_squared"], header=None, sep=" ", index_col=False)
    # Z = np.exp(scipy.special.logsumexp(df['log_weight']))
    # print(Z)
    df.drop(df.tail(1).index,inplace=True)
    df.drop(df.head(nlive).index,inplace=True)

    df['t'] = np.log(nlive/(nlive+1))
    df['logX'] = df['t'].cumsum()
    logXp = df['logX'].shift(1, fill_value=0)
    logXm = df['logX'].shift(-1, fill_value=-np.inf)
    df['logdX'] = np.log(1 - np.exp(logXm-logXp)) + logXp - np.log(2)

    # df['logW'] = df['logdX'] + df['log_like']
    df['logW'] = df['logdX']

    logZ = scipy.special.logsumexp(df['logW'])

    df['logW'] -= logZ

    plot_observables(df)
    # equal_mags = posterior_points(df['mag'], df['logW'])
    # equal_mags2 = posterior_points(df['mag']**2, df['logW'])
    # auto_corr = autocorrelation(np.abs(equal_mags))

    phi = np.average(df['mag'], weights = np.exp(df['logW']))
    abs_phi = np.average(np.abs(df['mag']), weights = np.exp(df['logW']))
    phi_squared = np.average(df['mag']**2, weights = np.exp(df['logW']))
    phi_4 = np.average(df['mag']**4, weights = np.exp(df['logW']))

    chi = params[2]**2 * (phi_squared - abs_phi**2)

    binder = 1 - phi_4 / (3 * phi_squared**2)

    results = {
        'kappa': params[0],
        'lambda': params[1],
        'mean_phi': phi,
        'mean_mod_phi': abs_phi,
        'phi_squared': phi_squared,
        'chi': chi,
        'U': binder,
        'iters': iters,
        # 'autocorrelation': auto_corr[0]
    }

    return results


def scaling_ansatz(x, a, b, c):
    return a + b * x**c


def residuals(c, x, y):
    # yfit = scaling_ansatz(x, c[0], c[1], c[2])
    # rs = (y - yfit)**2
    kappa_crit = y - c[0]
    ln_y_offset = np.log(y - c[0])
    ln_scaling_law = np.log(c[1]) + c[2] * np.log(x)

    rs = (ln_y_offset - ln_scaling_law)**2

    return rs.sum()

def load_scaling_data():
    scaling_dfs = {}

    # subfolder_list = os.listdir(root + scaling_folder)
    subfolder_list = ['32']
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
            params.append(int(n))

            out = read_file(path, params)
            data.append(out)

            files_searched.append(fname)

        scaling_data = pd.DataFrame(data).sort_values(by='kappa')

        scaling_dfs[int(n)] = scaling_data

    return scaling_dfs



def find_critical_point(data, n):
    kappa = np.linspace(data['kappa'].min(), data['kappa'].max(), 1000)
    if n == 20:
        s_val = 3
    elif n == 30:
        s_val = 1
    else:
        s_val = 0.1

    chi_fit = scipy.interpolate.UnivariateSpline(data['kappa'], data['chi'], k=3, s=s_val)  # Cubic spline

    idx = np.argmax(chi_fit(kappa))
    kappa_c = kappa[idx]
    chi_c = chi_fit(kappa)[idx]

    fig, ax = plt.subplots()
    ax.plot(data['kappa'], data['chi'], linestyle='', marker='x')
    # ax.plot(kappa, chi_fit(kappa), linestyle='--')
    ax.set_title('Chi Vs Kappa, n = {}'.format(n))

    # fig, ax = plt.subplots()
    # ax.plot(data['kappa'], data['autocorrelation'], linestyle='', marker='x')
    # ax.plot(kappa, chi_fit(kappa), linestyle='--')
    # ax.set_title('Autocorrelation Vs Kappa, n = {}'.format(n))

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
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Critical Kappa vs 1/L (inverse lattice size)')
    ax.set_xlabel('1 / L (lattice size)')
    ax.set_ylabel('Kappa critical')


def find_continuum_kappa(critical):
    coeff = np.polyfit(critical['inverse_n'], critical['kappa'], 1)
    return coeff[1]

def kappa_scaling_method2(critical):
    k0 = find_continuum_kappa(critical)
    y = np.log(critical['kappa'] - k0)

    slope_1, intercept_1, r_val_1, p_val_1, stderr_1 = scipy.stats.linregress(np.log(critical['n']), y)

    print(slope_1, intercept_1, r_val_1, p_val_1, stderr_1)

    fig, ax = plt.subplots()
    ax.plot(critical['inverse_n'], critical['kappa'], linestyle='', marker='x', label='Kappa Crit')

    x = np.linspace(critical['inverse_n'].min(), critical['inverse_n'].max() * 1.1)
    ax.plot(x, k0 + np.exp(intercept_1) * x**(-slope_1), linestyle = '--')

    # ax.plot(x, y, linestyle = '--')
    # ax.axvline(0, linestyle='--', color='black')
    ax.set_title('Critical Kappa vs 1/N (inverse lattice size)')
    ax.set_xlabel('1 / N (lattice size)')
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

    # sizes = np.array([40, 50, 60, 70, 80, 90])
    sizes = np.array([32])
    # sizes = [int(n) for n in os.listdir(root + scaling_folder)]

    critical = pd.DataFrame(columns=['kappa', 'chi', 'iters', 'n', 'inverse_n'])

    for n in sizes:
        data = scaling_data[n]

        kappa_c, chi_c = find_critical_point(data, n)
        critical.loc[len(critical)] = [kappa_c, chi_c, data['iters'].mean(), n, 1/n]


    # plot_kappa_scaling(critical)
    # plot_chi_scaling(critical)
    kappa_scaling_method2(critical)

    fig, ax = plt.subplots()
    ax.plot(critical['n'], critical['iters'], linestyle='', marker='x')

def phase_diagram(load = False):
    if load:
        phase_data = load_phase_data()
        phase_data.to_csv('phase_data.csv', index=False)
    phase_data = pd.read_csv('phase_data.csv')

    # table = phase_data.pivot('lambda', 'kappa', 'mean_mod_phi')
    table = phase_data.pivot_table(index='lambda', columns='kappa', values='mean_mod_phi')
    # interp_df = table.interpolate(method='linear', axis=1)

    ax = sns.heatmap(table, vmin=0, vmax=1, cmap='mako')
    # ax = sns.heatmap(interp_df, vmin=0, vmax=3)

    #ax.locator_params(axis='y', nbins=6)
    #ax.locator_params(axis='x', nbins=5)

    ax.invert_yaxis()
    ax.set_title(r"$\langle M \rangle$" + " Phase Diagram for 32x32 Lattice")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\lambda$")


def plot_phase_line(l):
    phase_data = pd.read_csv('phase_data.csv')

    phase_data = phase_data[phase_data['lambda'] == l]

    fig, ax = plt.subplots()
    ax.plot(phase_data['kappa'], phase_data['chi'])

# print(phase_data)

# scaling_analysis()
# phase_diagram()
# plot_phase_line(0.02)
# print(read_file(root + file, [0.35, 0.1]))
print(read_file(root + "Phi4_posterior_sampling", [0.26, 0.02, 32]))
# print(read_file(root + "scaling/32/Phi4_0.256000_0.020000", [0.256000, 0.02, 32]))
# print(read_file(root + "scaling/80/Phi4_0.200980_0.100000", [0.200350, 0.1]))
# print(read_file(root + "scaling/80/Phi4_0.200170_0.100000", [0.200170, 0.1]))

plt.show()
