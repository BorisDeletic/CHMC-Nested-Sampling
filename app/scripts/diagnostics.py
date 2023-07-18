import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/Phi4_posterior_weightings.diagnostics"

df = pd.read_csv(path)

df['reflections_MA'] = df['reflections'].rolling(window=500).mean()

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['reflections'] / df['steps'], marker = 'x')

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['epsilon'], marker = 'x')

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['accept_prob'], marker = 'x')

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['metric'], marker = 'x')
#
# fig, ax = plt.subplots()
# ax.scatter(df['iter'], df['epsilon'], marker = 'x')


print((df['accept_prob'] == 0).sum())
print((df['accept_prob'] == 1).sum())
print((df['reflections'] / df['steps'] > 0.99).sum())
print((df['accept_prob'] == 0) & (df['reflections'] / df['steps'] > 0.99).sum())



plt.show()