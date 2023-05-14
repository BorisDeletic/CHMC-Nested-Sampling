import anesthetic as ns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

def exp(x, a, m, c):
    return a * np.exp(-m * x) + c

#path = "/rds/user/bd418/hpc-work/correlation"
path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/phi4/correlation"

R = 128

file_list = sorted(os.listdir(path))
files_searched = []

print(file_list)

def correlationLength(correlations):

    log_correlations = np.log(np.abs(correlations[:len(correlations)//2]))

    print(log_correlations)

    xis = []
    kappas = []
    fig, ax = plt.subplots()
    for kappa in log_correlations.columns.values:
        m,b = np.polyfit(log_correlations.index.values, log_correlations[kappa], 1)

        xi = -1 / m

        xis.append(xi)
        kappas.append(kappa)
        ax.plot(log_correlations[kappa], label="k={:.5f}, xi={:.3f}".format(float(kappa), xi))
    #    ax.plot(kappa, , label="k={:.5f}, xi={:.3f}".format(kappa, xi))

    fig, ax1 = plt.subplots()
    ax1.scatter(kappas, xis)
    ax1.legend(loc="upper right")

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

        print("meanmag = {}".format(mean_mag))
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


#read_correlation_data()

correlation_samples = pd.read_csv("correlation_data.csv")
correlationLength(correlation_samples)


fig, ax = plt.subplots()

print(correlation_samples.columns.values)
kappa = '0.11748'
init_period = R//2
init_decay = correlation_samples[kappa][:init_period]
ax.plot(init_decay, label="k={:.6f}".format(float(kappa)))
kappa = '0.11758'
init_decay = correlation_samples[kappa][:init_period]
ax.plot(init_decay, label="k={:.6f}".format(float(kappa)))
kappa = '0.11754'
init_decay = correlation_samples[kappa][:init_period]
ax.plot(init_decay, label="k={:.6f}".format(float(kappa)))


#for kappa in correlation_samples.columns.values:
    # init_period = R//2
    # init_decay = correlation_samples[kappa][:init_period]
    # ax.plot(init_decay, label="k={:.6f}".format(float(kappa)))

        # Fit the function to the data
    # popt, pcov = curve_fit(exp, init_decay.index.values, init_decay.values)
    #
    # # Print the optimal parameters
    # a, m, c = popt[0], popt[1], popt[2]
    # print(a,m,c)
    # x = np.arange(0, init_period, 1)
    # y = exp(x, a, m, c)
   # ax.plot(x, y)





ax.set_title('Lattice = ' + str(R))
ax.legend(loc="upper right")
plt.show()
#ax.figure.savefig('correlations.png')



