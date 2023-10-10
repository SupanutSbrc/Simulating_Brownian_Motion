import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("EUR_USD Historical Data.csv")
new_df = df[['Date', 'Price']]

df['Date'] = df['Date'].str.strip()
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values(by='Date', ascending=True)

historical_range = df[df['Date'].between('01/01/2018', '12/31/2022')]

# Changes of price from the previous day, remove NaN values
delta = historical_range['Price'].pct_change().dropna()
fixed_driftparameter = delta.mean()
fixed_sqrtVparameter = delta.std()

# Price end of 2022
initial_p = df[df['Date'] == '12/30/2022']['Price'].values[0]
step = 1

dates = historical_range['Date']
forecast_start_date = dates.max() + pd.DateOffset(days=1)
forecast_end_date = pd.to_datetime('06/30/2023', format='%m/%d/%Y')
tsimulation = (forecast_end_date - forecast_start_date).days + 1 
print(tsimulation)
forecasted_data = []
for j in range(5):
    fixed_forecast = [initial_p]
    for i in range(tsimulation):
        if dates in historical_range['Date'].values:
            # Standard Brownian E(B(t))= 0 Var(B(t))=t
            # B(t)~N(0,t)
            std_brownian = np.random.normal(0, np.sqrt(step))
            # Brownian motion with drift X(t) = μt + σ(B(t))
            Xt = (fixed_driftparameter * step) + (fixed_sqrtVparameter * std_brownian)
            fixed_forecast.append(fixed_forecast[-1] + Xt)
        else:
            fixed_forecast.append(fixed_forecast[-1])
    fixed_forecast = fixed_forecast[1:]
    forecasted_data.append(fixed_forecast)
   
forecasted_dates = pd.date_range(start='01/03/2023', periods=tsimulation, freq='D')
forecasted_df = [pd.DataFrame({'Date': forecasted_dates, 'Price': forecast}) for forecast in forecasted_data]

real_data_2023 = df[df['Date'].between('01/01/2023', '06/30/2023')]
historical_range = pd.concat([historical_range, real_data_2023], ignore_index=True)

plt.figure(figsize=(12, 6))
plt.plot(historical_range['Date'], historical_range['Price'], label='Real Data (2018-2023)', color='blue')
for i in range(5):
    plt.plot(forecasted_df[i]['Date'], forecasted_df[i]['Price'], label=f'Simulation {i + 1}', alpha=0.7)

plt.xlabel('Date')
plt.ylabel('EURUSD')
plt.legend()
plt.grid(True)
plt.show()





