import anesthetic as ns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

path = "cmake-build-debug/app"

def save_magnetisations(path):
    data_path = os.path.join(path, 'phase_diagram')
    out_path = os.path.join(path, 'data.csv')
    file_list = os.listdir(data_path)
    files_searched = []

    if (os.path.exists(out_path) == False):
        empty = pd.DataFrame(columns=['kappa', 'lambda', 'mag'])
        empty.to_csv(out_path, index=False)
        print(empty)


    for file in file_list:
        fname = file[:16]

        if fname in files_searched:
            continue
        else:
            files_searched.append(fname)

        mag_data = pd.read_csv(out_path)

        params = file[5:16].split('_')
        params = [float(x) for x in params]

        if (params[0] in mag_data['kappa'].values) and (params[1] in mag_data['lambda'].values):
            continue

        chains = os.path.join(data_path, fname)
        samples = ns.read_chains(chains)
        posterior = samples.posterior_points()

        try:
            mag = abs(posterior['m'])
        except:
            print(params)

        mean_mag = mag.mean()

        data = pd.DataFrame({
            'kappa': params[0],
            'lambda': params[1],
            'mag': mean_mag
        }, index=[0])

        print(mag_data)
        mag_data = pd.concat([mag_data, data], axis=0)
        mag_data.to_csv(out_path, index=False)



save_magnetisations(path)

read_file = os.path.join(path, 'data.csv')
df = pd.read_csv(read_file)
print(read_file)
print(df)
#table = df.pivot('lambda', 'kappa', 'mag')

#ax = sns.heatmap(table, vmin=0, vmax=15)
#ax.invert_yaxis()
print(table)
#plt.show()



#samples.gui()
#plt.hist(abs(posterior['m']))
#plt.show()
