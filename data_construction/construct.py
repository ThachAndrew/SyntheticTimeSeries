import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import kpss, adfuller

NUM_SERIES =  1000
SERIES_LENGTH = 2000
INITIAL_SEGMENT_SIZE = 60
SEED = 55555
REGEN_SEED_MAX = 10 ** 8

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

        ar_t = e_series[i] + np.dot(reversed_phis, visible_series)

        ar.append(ar_t)

    ar = ar[p:]

    return ar

def generate_sar_phis(ar_phis, sar_phis, P, period):
    phis = np.zeros(max(len(ar_phis), P * period))
    phis[0:len(ar_phis)] = ar_phis
    for x in range(P):
        phis[((x + 1) * period) - 1] = sar_phis[x]

    return phis

def generate_multiple_series(num_series, length, p, P, period, seed, enforce_stationarity=True):
    series = np.empty((num_series, length))
    coefs = np.empty((num_series, max(p, P * period)))

    np.random.seed(seed)

    for x in range(num_series):
        print(x)
        ar_coefs = [(x - 0.5) * 2 for x in np.random.rand(p)]
        sar_coefs = [(x - 0.5) * 2 for x in np.random.rand(P)]

        all_coefs = generate_sar_phis(ar_coefs, sar_coefs, P, period)
        char_polynomial = np.append(1, all_coefs)[::-1]

        if enforce_stationarity:
            while np.min(np.abs(np.roots(char_polynomial))) < 1:
                ar_coefs = [(x - 0.5) * 2 for x in np.random.rand(p)]
                sar_coefs = [(x - 0.5) * 2 for x in np.random.rand(P)]

                all_coefs = generate_sar_phis(ar_coefs, sar_coefs, P, period)
                char_polynomial = np.append(1, all_coefs)[::-1]
        coefs[x] = all_coefs
        series[x]  = np.array([generate_ar(length, all_coefs)])

    return series, coefs

def main():
    P = 1
    p = 1
    period = 10

    np.random.seed(SEED)

    generated_series, coefs = generate_multiple_series(NUM_SERIES, SERIES_LENGTH, p, P, period, seed=SEED)
    variances = [np.var(series) for series in generated_series]

    for series_idx in range(NUM_SERIES):
        while np.isnan(variances[series_idx]) or np.isinf(variances[series_idx]):
            print("Regenerating series " + str(series_idx) + " due to infinite or undefined variance.")
            series_and_coefs = generate_multiple_series(1, SERIES_LENGTH, p, P, period, np.random.randint(0, REGEN_SEED_MAX))
            generated_series[series_idx] = series_and_coefs[0]
            coefs[series_idx] = series_and_coefs[1]
            variances[series_idx] = np.var(generated_series[series_idx])

    lower_var_thresh = np.quantile(variances, 0.2)

    series = [(generated_series[x], coefs[x], variances[x], x) for x in range(NUM_SERIES) if variances[x] <= lower_var_thresh]

    series.sort(key=lambda x: x[2])


    for gen_series in series[30:40]:
        print(adfuller(gen_series[0]))
        plt.plot(gen_series[0][100:200])
        plt.show()

if __name__ == '__main__':
    main()
