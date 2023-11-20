import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

def percentage_formatter(x, pos):
    return f'{x*100:.1f}%'


prior_width = 60
path_length = 500

path = '/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/tests/'
fname1 = 'gaussian_batch_nlive20D.csv'
fname2 = 'gaussian_batch_nlive1000.csv'

df1 = pd.read_csv(path + fname1)
df2 = pd.read_csv(path + fname2)

df1['like_calls'] = df1['iters'] * path_length
df2['like_calls'] = df2['iters'] * path_length

df1['error'] = abs((df1['logZ'] - df1['true_logZ']))
df2['error'] = abs((df2['logZ'] - df2['true_logZ']))

df1['error_pcnt'] = abs((df1['logZ'] - df1['true_logZ']) / df1['logZ'])
df2['error_pcnt'] = abs((df2['logZ'] - df2['true_logZ']) / df2['logZ'])


def get_lobf(ax, df):
    # Calculate the line of best fit in log-log space
    log_x = np.log10(df['dimension'])
    log_y = np.log10(df['like_calls'])

    coefficients = np.polyfit(log_x, log_y, 1)
    poly = np.poly1d(coefficients)

    # Create a range of x values for the line of best fit
    x_fit = np.linspace(min(log_x), max(log_x), 100)
    y_fit = poly(x_fit)

    return 10**x_fit, 10**y_fit, coefficients[0]


def plot_dim_vs_ncall():

    fig,ax = plt.subplots()

    xfit, yfit, m = get_lobf(ax, df1)
    ax.plot(xfit, yfit, label='nlive = 20D, m = {:.2f}'.format(m), linestyle='--')
    ax.scatter(df1['dimension'], df1['like_calls'], marker='x')

    xfit, yfit, m = get_lobf(ax, df2)
    ax.plot(xfit, yfit, label='nlive = 1000, m = {:.2f}'.format(m), linestyle='--')
    ax.scatter(df2['dimension'], df2['like_calls'], marker='x')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Likelihood calls')
    ax.legend()
    ax.set_title('Gaussian Likelihood - Dimension vs Number of likelihood calls')
    ax.grid(True)


def plot_evidence():
    fig, ax = plt.subplots()

    dim = np.linspace(min(df1['dimension']), max(df1['dimension']), 100)
    true_logZ = -dim * np.log(prior_width)

    ax.plot(dim, true_logZ, linestyle='--', label='Analytic Evidence')
    ax.plot(df1['dimension'], df1['logZ'], 'x', label='CHMC Evidence (LogZ)')
    ax.set_xscale('log')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('LogZ')
    ax.legend()
    ax.set_title('Gaussian Likelihood - Dimension vs Evidence (nlive = 20D)')


def plot_evidence_error():
    fig, (ax1, ax2) = plt.subplots(2)

    avg_error1 = df1.groupby('dimension')['error'].mean()
    avg_error2 = df2.groupby('dimension')['error'].mean()
    avg_error_pcnt1 = df1.groupby('dimension')['error_pcnt'].mean()
    avg_error_pcnt2 = df2.groupby('dimension')['error_pcnt'].mean()

    ax1.plot(avg_error1, '--x', label='nlive=20D')
    ax1.plot(avg_error2, '--x', label='nlive=1000')
    ax1.set_xscale('log')
    ax1.set_ylabel('Avg LogZ Error (absolute)')
    ax1.set_title('Gaussian Likelihood - Dimension vs Evidence error')
    ax1.legend()
    ax1.grid(True)

    ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax2.plot(avg_error_pcnt1, '--x', label='nlive=20D')
    ax2.plot(avg_error_pcnt2, '--x', label='nlive=1000')
    ax2.set_xscale('log')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Avg LogZ Error (%)')
    ax2.legend()
    ax2.grid(True)


plot_dim_vs_ncall()
plot_evidence()
plot_evidence_error()

plt.show()

