import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("EUR_USD Historical Data.csv.")
new_df = df[['Date', 'Price']]

df['Date'] = df['Date'].str.strip()
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values(by='Date', ascending=True)

################ Fixed parameter forecasting (2018-2022) ################

historical_range = df[df['Date'].between('01/01/2018', '12/31/2022')]
# Changes of price from the previous day, remove NaN values
delta = historical_range['Price'].pct_change().dropna()
fixed_driftparameter = delta.mean()
fixed_sqrtVparameter = delta.std()

# Price at the end of 2022
initial_p = df[df['Date'] == '12/30/2022']['Price'].values[0]
step = 1
tsimulation = df[df['Date'].between('01/03/2023', '6/30/2023')].shape[0]
fixed_forecast = [initial_p]

for i in range(tsimulation):
    # Standard Brownian motion E(B(t))= 0 Var(B(t))=t
    # B(t)~N(0, t)
    std_brownian = np.random.normal(0, np.sqrt(step))
    # Brownian motion with drift X(t) = μt + σ(B(t))
    Xt = (fixed_driftparameter * step) + (fixed_sqrtVparameter * std_brownian)
    fixed_forecast.append(fixed_forecast[-1] + Xt)
fixed_forecast.remove(fixed_forecast[0])

################ Moving parameter forecasting ################

current_driftparameter = fixed_driftparameter
current_sqrtVparameter = fixed_sqrtVparameter
true_price_index = df[df['Date'] == '12/30/2022'].index[0]
current_p = df['Price'][true_price_index]

moving_forecast = [current_p]
rowstart = df[df['Date'] == '01/02/2018'].index[0]
rowend = df[df['Date'] == '12/30/2022'].index[0]

for i in range(tsimulation):
    historical_range = df[df['Date'].between(df['Date'][rowstart], df['Date'][rowend])]
    moving_delta = historical_range['Price'].pct_change().dropna()
    std_brownian = np.random.normal(0, np.sqrt(step))
    Xt = (current_driftparameter * step) + (current_sqrtVparameter * std_brownian)
    moving_forecast.append(current_p + Xt)
    rowstart -= 1
    rowend -= 1
    current_driftparameter = moving_delta.mean()
    current_sqrtVparameter = moving_delta.std()
    true_price_index -= 1
    current_p = df['Price'][true_price_index]
moving_forecast.remove(moving_forecast[0])

true_value = df[df['Date'].between('1/3/2023', '6/30/2023')]

plt.figure(figsize=(12, 6))
plt.plot(true_value['Date'], fixed_forecast, label='Fixed-Parameter Forecast', linestyle='-')
plt.plot(true_value['Date'], moving_forecast, label='Moving-Parameter Forecast', linestyle='-')
plt.plot(true_value['Date'], true_value['Price'], label='Actual Data', linestyle='-', color='black', linewidth=2)
plt.xlabel('Date')
plt.ylabel('EURUSD')
plt.grid(True)
plt.legend()
plt.show()





