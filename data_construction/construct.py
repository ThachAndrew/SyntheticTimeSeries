import matplotlib.pyplot as plt
import numpy as np

def generate_wn(n, sigma=1):
    return np.random.normal(0, sigma, size=n)

def generate_ar(n, phis, sigma=1):
    p = len(phis)
    adj_n = n + p
    e_series = generate_wn(adj_n, sigma)

    ar = [e_series[0]]
    for i in range(1, adj_n):
        visible_phis = phis[0:min(p, i)]
        visible_series = ar[i - min(p, i):i]

        reversed_phis = visible_phis[::-1]

        ar_t = max(0, e_series[i] + np.dot(reversed_phis, visible_series))

        ar.append(ar_t)

    ar = ar[p:]

    return ar

def generate_sar_phis(ar_phis, sar_phis, P, period):
    phis = np.zeros(P * period)
    phis[0:len(ar_phis)] = ar_phis
    for x in range(P):
        phis[((x + 1) * period) - 1] = sar_phis[x]

    return phis

def generate_multiple_series(num_series, length, p, P, period):
    series = np.empty((num_series, length))

    for x in range(num_series):
        ar_coefs = [(x - 0.5) * 2 for x in np.random.rand(p)]
        sar_coefs = [(x - 0.5) * 2 for x in np.random.rand(P)]

        series[x]  = np.array([generate_ar(length, generate_sar_phis(ar_coefs, sar_coefs, P, period))])

    return series

def main():
    num_series = 500
    length = 600
    p = 2
    P = 2
    period = 6

    generated_series = generate_multiple_series(num_series, length, p, P, period)

    variances = [np.var(series) for series in generated_series if np.var(series) < 1]

    plt.hist(variances)
    plt.show()


if __name__ == '__main__':
    main()