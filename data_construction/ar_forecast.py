import numpy as np

# Perform AR forecasting
def ar_forecast(x, coefs, bias, n):
    for i in range(n):
        x = np.append(x, np.dot(x[-len(coefs):], coefs[::-1]) + bias)

    return x[-n:]
