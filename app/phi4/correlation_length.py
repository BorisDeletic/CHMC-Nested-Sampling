import anesthetic as ns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os



path = "/rds/user/bd418/hpc-work/correlation"

R = 512
file_list = sorted(os.listdir(path))
files_searched = []

print(file_list)
correlation_samples = pd.DataFrame()

def correlationLength(correlations):
    #split half way as symmetric about L/2
    log_correlations = np.log(np.abs(correlations))

    #print(log_correlations)

    xis = []
    kappas = []
    fig, ax = plt.subplots()

    for kappa in log_correlations.columns.values:
        front_x = log_correlations.index.values[:R//10]
        front_y = log_correlations[kappa][:R//10]

        m,b = np.polyfit(front_x, front_y, 1)

        xi = -1 / m

        xis.append(xi)
        kappas.append(kappa)
        ax.plot(log_correlations[kappa], label="k={:.6f}, xi={:.1f}".format(kappa, xi))
    #    ax.plot(kappa, , label="k={:.5f}, xi={:.3f}".format(kappa, xi))
    
    ax.set_title("Log Correlation Func vs R")
    #ax.scatter(kappas, xis)
    ax.legend(loc="upper right")

    ax.figure.savefig('/rds/user/bd418/hpc-work/images/log_corr.png', dpi=300)


fig, ax = plt.subplots()
for file in file_list:
    fname = file[:22]

    if fname in files_searched:
        continue
    else:
        files_searched.append(fname)

    kappa = float(fname.split("_")[1])
    l = fname.split("_")[2]

    try:
        chains = os.path.join(path, fname)
        samples = ns.read_chains(chains)
    except:
        continue

    posterior = samples.posterior_points()

    mean_mag = abs(posterior['mag']).mean()

    print("meanmag = {}".format(mean_mag))
    correlations = []
    for r in range(1, R):
        c_key = "c_{}".format(r)

        mean_correlation = posterior[c_key].mean() - mean_mag*mean_mag
       # print(c_key)
        #print(mean_correlation)
        correlations.append(mean_correlation)

    correlations /= posterior["c_0"].mean()
    correlation_samples[kappa] = correlations

    ax.plot(correlations, label="k={:.6f}, m={:.2f}".format(kappa, mean_mag))

#print(correlation_samples[:32])

correlationLength(correlation_samples)

ax.set_title("Correlation Function vs R")
ax.legend(loc="upper right")
#plt.show()
ax.figure.savefig('/rds/user/bd418/hpc-work/images/correlations.png', dpi=300)



