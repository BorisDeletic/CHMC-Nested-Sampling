import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/"
# chain = "gaussian_200d_100nlive_r1.rejected_points"
chain = "Phi4_posterior_sampling.rejected_points"

with open(path + chain) as f:
    headers = f.readline().split(",")
    print(headers)
    for i in range(40):
        f.readline()
        # skip to 10th point
    # print(stats)
    stats_df = pd.read_csv(io.StringIO(f.readline()), names = headers)

    deltaX = np.fromstring(f.readline()[:-2], sep=',')

    likes = np.fromstring(f.readline()[:-2], sep=',')

    energies = np.fromstring(f.readline()[:-2], sep=',')


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(stats_df)

fig, ax = plt.subplots()
ax.plot(deltaX)
ax.set_title("delta x vs integration step")

fig, ax = plt.subplots()
ax.plot(likes)
ax.set_title("likelihood vs integration step")
ax.axhline(stats_df['birth_likelihood'].iloc[0], linestyle='--')

fig, ax = plt.subplots()
ax.plot(energies)
ax.set_title("energy vs integration step")

plt.show()