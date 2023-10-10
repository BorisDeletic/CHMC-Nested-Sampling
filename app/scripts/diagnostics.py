import pandas as pd
import matplotlib.pyplot as plt



path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/Phi4_posterior_weightings.diagnostics"

df = pd.read_csv(path)
df = df[df['rejected'] == 0]

df['likelihood_increase_MA'] = (df['birth_likelihood'] - df['likelihood']).rolling(window=500).mean()
df['reflections_MA'] = df['reflections'].rolling(window=500).mean()
df['reflections_1step'] = df[df['steps'] == 1]['reflections']
df['reflections_2step'] = df[df['steps'] == 2]['reflections']
df['reflections_3step'] = df[df['steps'] == 3]['reflections']

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['reflections'] / df['steps'], marker = 'x')
ax.scatter(df['iter'], df['reflections_1step'] / df['steps'], marker = 'x', label='path length 1')
ax.scatter(df['iter'], df['reflections_2step'] / df['steps'], marker = 'x', label='path length 2')
ax.scatter(df['iter'], df['reflections_3step'] / df['steps'], marker = 'x', label='path length 3')
ax.set_title('reflect rate')

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['epsilon'], marker = 'x')
ax.set_title('epsilon')

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['accept_prob'], marker = 'x')
ax.set_title('accept probability')

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['metric'], marker = 'x')
ax.set_title('metric')
#
fig, ax = plt.subplots()
ax.scatter(df['iter'], df['steps'], marker = 'x')
ax.set_title('steps')

fig, ax = plt.subplots()
ax.scatter(df['iter'], df['reflections'], marker = 'x')
ax.set_title('reflections')


fig, ax = plt.subplots()
ax.plot(df['iter'], df['birth_likelihood'] - df['likelihood'], marker = 'x', linestyle='')
ax.plot(df['iter'], df['likelihood_increase_MA'], linestyle = '-.')
# ax.scatter(df['iter'], df['birth_likelihood'], marker = 'x')
ax.set_title('log birth likelihood - new likelihood')


fig, ax = plt.subplots()
ax.plot(df['iter'], df['birth_likelihood'], marker = 'x', linestyle='')
# ax.scatter(df['iter'], df['birth_likelihood'], marker = 'x')
ax.set_title('log birth likelihood')

# fig, ax = plt.subplots()
# ax.plot(df['iter'], df['accept_prob'].cumsum()/(df['iter']+1), marker = 'x', linestyle='')
# ax.set_title('average accept prob')

# fig, ax = plt.subplots()
# ax.scatter(df['iter'], df['epsilon'], marker = 'x')


print("zero_accept_prob={}".format((df['accept_prob'] == 0).sum()))
print("one_accept_prob={}".format((df['accept_prob'] == 1).sum()))
print("reflection_hops={}".format((df['reflections'] / df['steps'] > 0.99).sum()))
# print((df['accept_prob'] == 0) & (df['reflections'] / df['steps'] > 0.99).sum())
print(df['iter'])


plt.show()