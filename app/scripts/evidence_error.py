import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

def percentage_formatter(x, pos):
    return f'{x*100:.1f}%'


prior_width = 60
path = '/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/tests/gaussian_batch.csv'

df = pd.read_csv(path)

def plot_dim_vs_ncall():
    path_length = 500
    df['like_calls'] = df['iters'] * path_length

    # Calculate the line of best fit in log-log space
    log_x = np.log10(df['dimension'])
    log_y = np.log10(df['like_calls'])
    coefficients = np.polyfit(log_x, log_y, 1)
    poly = np.poly1d(coefficients)

    # Create a range of x values for the line of best fit
    x_fit = np.linspace(min(log_x), max(log_x), 100)
    y_fit = poly(x_fit)

    fig,ax = plt.subplots()

    # Plot the data points and the line of best fit on a log-log scale
    # plt.figure(figsize=(8, 6))
    ax.plot(10**x_fit, 10**y_fit, 'r', label='Line of Best Fit, m = {:.2f}'.format(coefficients[0]), linestyle='--')
    ax.scatter(df['dimension'], df['like_calls'], label='Data')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Likelihood calls')
    ax.legend()
    ax.set_title('Gaussian Likelihood - Dimension vs Number of likelihood calls')
    ax.grid(True)


def plot_evidence():
    fig, ax = plt.subplots()

    dim = np.linspace(min(df['dimension']), max(df['dimension']), 100)
    true_logZ = -dim * np.log(prior_width)

    ax.plot(dim, true_logZ, linestyle='--', label='Analytic Evidence')
    ax.plot(df['dimension'], df['logZ'], 'x', label='CHMC Evidence (LogZ)')
    ax.set_xscale('log')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('LogZ')
    ax.legend()
    ax.set_title('Gaussian Likelihood - Dimension vs Evidence (nlive = 20D)')


def plot_evidence_error():
    fig, (ax1, ax2) = plt.subplots(2)

    df['error'] = abs((df['logZ'] - df['true_logZ']))
    df['error_pcnt'] = abs((df['logZ'] - df['true_logZ']) / df['logZ'])

    avg_error = df.groupby('dimension')['error'].mean()
    avg_error_pcnt = df.groupby('dimension')['error_pcnt'].mean()

    ax1.plot(avg_error, 'x', label='Avg Evidence Error (absolute)')
    ax1.set_xscale('log')
    ax1.set_ylabel('Avg LogZ Error')
    ax1.set_title('Gaussian Likelihood - Dimension vs Evidence error (nlive = 20D)')
    ax1.legend()
    ax1.grid(True)

    ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax2.plot(avg_error_pcnt, 'x', label='Avg Evidence Error (%)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Avg LogZ Error (%)')
    ax2.legend()
    ax2.grid(True)


plot_dim_vs_ncall()
plot_evidence()
plot_evidence_error()

plt.show()

