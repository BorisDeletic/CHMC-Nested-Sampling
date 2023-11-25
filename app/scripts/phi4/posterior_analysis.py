import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os

# can get this number from .stats
nlive = 500
n=32

#root = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/"
root="/home/bd418/rds/hpc-work/"
phase_folder = "phase_diagram/"
scaling_folder = "scaling/"
file = "Phi4_posterior_sampling"


def read_file(fname, params):
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

    binder = 1 - phi_4 / (3 * phi_squared**2)
    # print(chi)
    results = {
        'kappa': params[0],
        'lambda': params[1],
        'mean_phi': phi,
        'mean_mod_phi': abs_phi,
        'phi_squared': phi_squared,
        'chi': chi,
        'U': binder
    }

    return results


def load_phase_data():
    data = []

    file_list = os.listdir(root + phase_folder)
    for file in file_list:
        fname = file[:20]

        params = file[5:20].split('_')
        params = [float(x) for x in params]

        out = read_file(root + phase_folder + fname, params)
        data.append(out)


    phase_data = pd.DataFrame(data).sort_values(by='kappa')

    return phase_data


def load_scaling_data():
    scaling_dfs = {}

    subfolder_list = os.listdir(root + scaling_folder)
    for n in subfolder_list:
        data = []
        file_list = os.listdir(root + scaling_folder + n)

        for file in file_list:
            fname = file[:20]
            path = root + scaling_folder + n + "/" + fname

            params = file[5:20].split('_')
            params = [float(x) for x in params]

            out = read_file(path, params)
            data.append(out)


        scaling_data = pd.DataFrame(data).sort_values(by='kappa')

        scaling_dfs[int(n)] = scaling_data

    return scaling_dfs



scaling_data = load_scaling_data()
print(scaling_data)

# phase_data = load_phase_data()
# print(phase_data)

sizes = [16,32,64,128]
for n in sizes:
	data = scaling_data[n]
	fig, ax = plt.subplots()
	ax.plot(data['kappa'], data['chi'], linestyle='', marker='x')
	plt.savefig(root + "susceptibility_" + str(n) + ".png")

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


#plt.show()
#plt.savefig(root + "susceptibility_" + n + ".png")
