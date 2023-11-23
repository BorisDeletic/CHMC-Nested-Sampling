import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os

nlive = 500
n=32
logZ = 725.957

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

root = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/"
phase_folder = "phase_diagram/"
file = "Phi4_posterior_sampling"

data = []

# out = read_file(root + file, [0.25, 0.02])
# data.append(out)
# print(data)

file_list = os.listdir(root + phase_folder)
for file in file_list:
    fname = file[:20]

    params = file[5:20].split('_')
    params = [float(x) for x in params]

    out = read_file(root + phase_folder + fname, params)
    data.append(out)


phase_data = pd.DataFrame(data).sort_values(by='kappa')

print(phase_data)

fig, ax = plt.subplots()
ax.plot(phase_data['kappa'], phase_data['chi'], linestyle='', marker='x')
#
fig, ax = plt.subplots()
ax.plot(phase_data['kappa'], phase_data['U'], linestyle='', marker='x')

fig, ax = plt.subplots()
ax.plot(phase_data['kappa'], phase_data['mean_mod_phi'], linestyle='', marker='x')
#
# fig, ax = plt.subplots()
# ax.hist(df['mag'], weights=df['weight'], bins=100)


plt.show()
