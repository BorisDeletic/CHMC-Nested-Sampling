import anesthetic as ns
import matplotlib.pyplot as plt
import pandas as pd
import os


path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/phi4/correlation"
#file = "Phi4_0.16_0.50"

file_list = os.listdir(path)
files_searched = []

print(file_list)
for file in file_list:
    fname = file[:20]

    if fname in files_searched:
        continue
    else:
        files_searched.append(fname)

    kappa = fname.split("_")[1]
    l = fname.split("_")[2]

    chains = os.path.join(path, fname)
    samples = ns.read_chains(chains)
    posterior = samples.posterior_points()

    correlations = [posterior["c_0"].mean()]
    for r in range(1, 15):
        c_key = "c_{}".format(r)

        mean_correlation = posterior[c_key].mean() - correlations[0]
        print(c_key)
        print(mean_correlation)

        correlations.append(mean_correlation)

    plt.plot(correlations, label="kappa={}".format(kappa))


#samples.gui()
#print(samples)
plt.legend(loc="upper left")
plt.show()
