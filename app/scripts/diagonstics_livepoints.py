import pandas as pd
import matplotlib.pyplot as plt



path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/Phi4_posterior_weightings.live_points"
numLive = 100

df = pd.read_csv(path)

df['reseeded'] = False
df = df[df['steps'] != -1] #remove all the initial points

df['reflect_rate'] = df['reflections'] / df['steps']


fig, ax = plt.subplots()
ax.plot(df['reflections'], linestyle = '', marker='x')
ax.set_title('reflections')
#
fig, ax = plt.subplots()

lifetime = 10 * numLive
for i in range(1000,df['iter'].max(),numLive):

    current_df = df[df['iter'] == i]
    last_df = df[df['iter'] == i - lifetime]

    df.loc[df['iter'] == i, 'reseeded'] = ~current_df['ID'].isin(last_df['ID'])



ax.plot(df[df['reseeded'] == True]['reflect_rate'], linestyle = '', marker='x', label='reseeded')
ax.plot(df[df['reseeded'] == False]['reflect_rate'], linestyle = '', marker='x', label='not reseeded')
ax.set_title('reflect rate, reseed lifetime={} iters'.format(lifetime))
ax.legend(loc='upper right')


plt.show()
