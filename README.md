
# Intro
Autoregressive models on time series data can be a challenging when it comes to complex distribution shift dynamics and noise.

This repo features a synthetic AR (autoregressive) generator, where we can control many parameters of the AR model.
It basically allows us to simulate an AR process.
By using this, one can introduce noise and/or control other parameters, of a time-series dataset.

Using a synthetic dataset helps evaluate different machine learned models for noise and dynamic distribution shifts in time series data.  

# Usage
The provided Jupyter notebook helps the user visualize parameter tweaks.  The last cell should be the one that actually generates the synthetic data file. 
In an ARIMA model there are 3 parameters that are used to help model the major aspects of a times series: seasonality, trend, and noise. These parameters are labeled linear_trend, seasonality, white_noise, weekday_factor. 

# Useful links

<https://otexts.com/fpp2/arima.html>

<https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c>

<https://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/>

<https://towardsdatascience.com/creating-synthetic-time-series-data-67223ff08e34>

<https://pypi.org/project/timeseries-generator/>




