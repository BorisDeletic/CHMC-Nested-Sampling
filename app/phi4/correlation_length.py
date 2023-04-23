import anesthetic as ns
import matplotlib.pyplot as plt
import pandas as pd
import os


path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/phi4/correlation"
#file = "Phi4_0.16_0.50"

file_list = sorted(os.listdir(path))
files_searched = []

print(file_list)
for file in file_list:
    fname = file[:21]

    if fname in files_searched:
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
    correlations = []
    for r in range(1, 30):
        c_key = "c_{}".format(r)

        mean_correlation = posterior[c_key].mean() #- mean_mag*mean_mag
        print(c_key)
        print(mean_correlation)

        correlations.append(mean_correlation)


    #correlations /= posterior["c_0"].mean()

    plt.axhline(mean_mag)
    plt.plot(correlations, label="k={:.3f}, m={:.2f}".format(kappa, mean_mag))


#samples.gui()
#print(samples)
plt.legend(loc="upper right")
plt.show()
