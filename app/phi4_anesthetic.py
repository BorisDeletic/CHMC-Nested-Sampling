import anesthetic as ns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

path = "/home/bd418/data"

def save_magnetisations(path):
    data_path = os.path.join(path, 'phase_diagram')
    file_list = os.listdir(data_path)
    files_searched = []
    phase_data = []

    for file in file_list:
        fname = file[:16]

        if fname in files_searched:
            continue
        else:
            files_searched.append(fname)

        params = file[5:16].split('_')
        params = [float(x) for x in params]

        chains = os.path.join(data_path, fname)
        samples = ns.read_chains(chains)
        posterior = samples.posterior_points()

        try:
            mag = abs(posterior['m'])
        except:
            print(params)

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

table = df.pivot('lambda', 'kappa', 'mag')

#ax = sns.heatmap(table, vmin=0, vmax=15)
#ax.invert_yaxis()
print(table)
#plt.show()



#samples.gui()
#plt.hist(abs(posterior['m']))
#plt.show()
