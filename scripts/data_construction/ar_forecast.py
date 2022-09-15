import numpy as np

# Perform AR forecasting
def ar_forecast(x, coefs, bias, n):
    for i in range(n):
        x = np.append(x, np.dot(x[-len(coefs):], coefs[::-1]) + bias)

    return x[-n:]

def cluster_ar_forecast_adjust(forecasts, sum):
    base_forecast_sum = np.sum(forecasts)
    adj_term = (sum - base_forecast_sum)/len(forecasts)

    return [forecast + adj_term for forecast in forecasts]

# Bias AR forecasts so their mean is equal to some specified value
def top_down_adjust_ar_forecast(forecast, mean):
    forecast_mean = np.mean(forecast)
    adj_bias = mean - forecast_mean

    return forecast + adj_bias

# Adjust forecasts to be coherent based on forecast proportions
def fp_adjust_ar_forecast(series, agg_series):
    coherent_forecasts = [[] for x in range(len(series))]

    for t in range(len(agg_series)):
        series_sum = sum(series[x][t] for x in range(len(series)))

        for s in range(len(series)):
            coherent_forecasts[s] += [agg_series[t] * (series[s][t] / series_sum)]

    print(agg_series)
    print(coherent_forecasts)

    return coherent_forecasts
