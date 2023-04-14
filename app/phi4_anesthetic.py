import anesthetic as ns
import anesthetic.convert
import matplotlib.pyplot as plt
import pandas as pd
import os

path = "cmake-build-debug/app/phase_diagram"

def save_magnetisations(path):
    phase_data = []
    files = os.listdir(path)

    for file in files:
        fname = file[:16]
        params = file[5:16].split('_')
        params = [float(x) for x in params]

        chains = os.path.join(path, fname)
        samples = ns.read_chains(chains)
        posterior = samples.posterior_points()

        mag = posterior['m']
        mean_mag = mag.mean()

        data = {
            'kappa': params[0],
            'lambda': params[1],
            'mag': mean_mag
        }

        phase_data.append(data)

        df = pd.DataFrame(phase_data)

        save_file = os.path.join(path, 'data.csv')
        df.to_csv(save_file)



save_magnetisations(path)

read_file = os.path.join(path, 'data.csv')
df = pd.read_csv(read_file)

#samples = ns.read_chains("cmake-build-debug/app/phase_diagram/Phi4_0.000_0.004")
#posterior = samples.posterior_points()

#dist = ns.convert.to_getdist(samples)


#samples.gui()
#plt.hist(abs(posterior['m']))
#plt.show()