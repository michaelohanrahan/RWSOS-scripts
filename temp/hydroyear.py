import pandas as pd 
import matplotlib.pyplot as plt


df = pd.read_csv(r"p:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202303\run_default\output.csv", index_col=0, parse_dates=True)

#%%
import matplotlib.pyplot as plt
import numpy as np
import calendar

BH_Q = df['Q_16']

# Group by month and calculate mean and standard deviation
monthly_mean = BH_Q.groupby(BH_Q.index.month).mean()
monthly_std = BH_Q.groupby(BH_Q.index.month).std()
# Find the month with the minimum mean flow
min_month = monthly_mean.idxmin()

# Create a new figure and plot the mean discharge with a legend
plt.figure(figsize=(12, 6))
plt.plot(monthly_mean.index, monthly_mean, label='Mean Discharge')

# Add a filled area representing the uncertainty (mean ± std dev)
plt.fill_between(monthly_mean.index, (monthly_mean - monthly_std), (monthly_mean + monthly_std), color='b', alpha=.1)

# Add a vertical line at the month with the minimum mean flow
plt.axvline(x=min_month, color='r', linestyle='--', label=f'Lowest Mean ({calendar.month_name[min_month]}): {monthly_mean[min_month]:.2f} ± {monthly_std[min_month]:.2f} (m³/s)')

# Filter the data for the first year up until August, group by month, calculate the mean, and plot it
first_year = BH_Q[BH_Q.index.year == BH_Q.index.year.min()]
first_year_until_august = first_year[first_year.index.month <= 8]
first_year_monthly_mean = first_year_until_august.groupby(first_year_until_august.index.month).mean()
plt.plot(first_year_monthly_mean.index, first_year_monthly_mean, 'k:', label='First year until August')

# Set x-ticks to month names
plt.xticks(monthly_mean.index, [calendar.month_name[i] for i in monthly_mean.index])

plt.xlabel('Time (month of the year)')
plt.ylabel('Discharge (m³/s)')
plt.title('Annual Monthly Discharge')
plt.legend(loc='upper right')
plt.savefig(r"p:\11209265-grade2023\wflow\wflow_meuse_julia\hydro_year.png")