import numpy as np

# Perform AR forecasting
def ar_forecast(x, coefs, bias, n):
    for i in range(n):
        x = np.append(x, np.dot(x[-len(coefs):], coefs[::-1]) + bias)

    return x[-n:]

def top_down_adjust_ar_forecast(forecast, mean):
    forecast_mean = np.mean(forecast)
    adj_bias = mean - forecast_mean

    return forecast + adj_bias
