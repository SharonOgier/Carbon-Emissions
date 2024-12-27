#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the datasets using your actual file paths
temperature_data = pd.read_csv(r'C:\Users\Sharon\Desktop\Docs\Carbon Emissions\temperature.csv')
co2_data = pd.read_csv(r'C:\Users\Sharon\Desktop\Docs\Carbon Emissions\carbon_emmission.csv')

# Preview the first few rows of each dataset
temperature_data_preview = temperature_data.head()
co2_data_preview = co2_data.head()

temperature_data_preview, co2_data_preview


# In[2]:


# selecting and computing statistics for temperature changes
temperature_values = temperature_data.filter(regex='^F').stack()  # extracting all year columns
temperature_stats = {
    "Mean": temperature_values.mean(),
    "Median": temperature_values.median(),
    "Variance": temperature_values.var()
}

# computing statistics for CO2 concentrations
co2_values = co2_data["Value"]  # extracting the Value column
co2_stats = {
    "Mean": co2_values.mean(),
    "Median": co2_values.median(),
    "Variance": co2_values.var()
}

temperature_stats, co2_stats


# In[3]:


import plotly.graph_objects as go
import plotly.express as px

# extracting time-series data for plotting
# temperature: averaging across countries for each year
temperature_years = temperature_data.filter(regex='^F').mean(axis=0)
temperature_years.index = temperature_years.index.str.replace('F', '').astype(int)

# CO2: parsing year and averaging monthly data
co2_data['Year'] = co2_data['Date'].str[:4].astype(int)
co2_yearly = co2_data.groupby('Year')['Value'].mean()

# time-series plot for temperature and CO2 levels
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=temperature_years.index, y=temperature_years.values,
    mode='lines+markers', name="Temperature Change (°C)"
))
fig.add_trace(go.Scatter(
    x=co2_yearly.index, y=co2_yearly.values,
    mode='lines+markers', name="CO₂ Concentration (ppm)", line=dict(dash='dash')
))
fig.update_layout(
    title="Time-series of Temperature Change and CO₂ Concentrations",
    xaxis_title="Year",
    yaxis_title="Values",
    template="plotly_white",
    legend_title="Metrics"
)
fig.show()

# correlation heatmap
merged_data = pd.DataFrame({
    "Temperature Change": temperature_years,
    "CO₂ Concentration": co2_yearly
}).dropna()

heatmap_fig = px.imshow(
    merged_data.corr(),
    text_auto=".2f",
    color_continuous_scale="RdBu",  # diverging colormap similar to coolwarm
    title="Correlation Heatmap"
)
heatmap_fig.update_layout(
    template="plotly_white"
)
heatmap_fig.show()

# scatter plot: temperature vs CO2 concentrations
scatter_fig = px.scatter(
    merged_data,
    x="CO₂ Concentration", y="Temperature Change",
    labels={"CO₂ Concentration": "CO₂ Concentration (ppm)", "Temperature Change": "Temperature Change (°C)"},
    title="Temperature Change vs CO₂ Concentration",
    template="plotly_white"
)
scatter_fig.update_traces(marker=dict(size=10, opacity=0.7))
scatter_fig.show()


# In[4]:


from scipy.stats import linregress

# temperature trend
temp_trend = linregress(temperature_years.index, temperature_years.values)
temp_trend_line = temp_trend.slope * temperature_years.index + temp_trend.intercept

# CO2 trend
co2_trend = linregress(co2_yearly.index, co2_yearly.values)
co2_trend_line = co2_trend.slope * co2_yearly.index + co2_trend.intercept

fig_trends = go.Figure()

fig_trends.add_trace(go.Scatter(
    x=temperature_years.index, y=temperature_years.values,
    mode='lines+markers', name="Temperature Change (°C)"
))
fig_trends.add_trace(go.Scatter(
    x=temperature_years.index, y=temp_trend_line,
    mode='lines', name=f"Temperature Trend (Slope: {temp_trend.slope:.2f})", line=dict(dash='dash')
))
fig_trends.add_trace(go.Scatter(
    x=co2_yearly.index, y=co2_yearly.values,
    mode='lines+markers', name="CO₂ Concentration (ppm)"
))
fig_trends.add_trace(go.Scatter(
    x=co2_yearly.index, y=co2_trend_line,
    mode='lines', name=f"CO₂ Trend (Slope: {co2_trend.slope:.2f})", line=dict(dash='dash')
))

fig_trends.update_layout(
    title="Trends in Temperature Change and CO₂ Concentrations",
    xaxis_title="Year",
    yaxis_title="Values",
    template="plotly_white",
    legend_title="Metrics"
)
fig_trends.show()

# seasonal variations in CO2 concentrations
co2_data['Month'] = co2_data['Date'].str[-2:].astype(int)
co2_monthly = co2_data.groupby('Month')['Value'].mean()

fig_seasonal = px.line(
    co2_monthly,
    x=co2_monthly.index,
    y=co2_monthly.values,
    labels={"x": "Month", "y": "CO₂ Concentration (ppm)"},
    title="Seasonal Variations in CO₂ Concentrations",
    markers=True
)
fig_seasonal.update_layout(
    xaxis=dict(tickmode="array", tickvals=list(range(1, 13))),
    template="plotly_white"
)
fig_seasonal.show()


# In[5]:


from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests

# pearson and spearman correlation coefficients
pearson_corr, _ = pearsonr(merged_data["CO₂ Concentration"], merged_data["Temperature Change"])
spearman_corr, _ = spearmanr(merged_data["CO₂ Concentration"], merged_data["Temperature Change"])

# granger causality test
granger_data = merged_data.diff().dropna()  # first differencing to make data stationary
granger_results = grangercausalitytests(granger_data, maxlag=3, verbose=False)

# extracting p-values for causality
granger_p_values = {f"Lag {lag}": round(results[0]['ssr_chi2test'][1], 4)
                    for lag, results in granger_results.items()}

pearson_corr, spearman_corr, granger_p_values


# In[6]:


import statsmodels.api as sm

# creating lagged CO2 data to investigate lagged effects
merged_data['CO₂ Lag 1'] = merged_data["CO₂ Concentration"].shift(1)
merged_data['CO₂ Lag 2'] = merged_data["CO₂ Concentration"].shift(2)
merged_data['CO₂ Lag 3'] = merged_data["CO₂ Concentration"].shift(3)

# dropping rows with NaN due to lags
lagged_data = merged_data.dropna()

X = lagged_data[['CO₂ Concentration', 'CO₂ Lag 1', 'CO₂ Lag 2', 'CO₂ Lag 3']]
y = lagged_data['Temperature Change']
X = sm.add_constant(X)  # adding a constant for intercept

model = sm.OLS(y, X).fit()

model_summary = model.summary()
model_summary


# In[7]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# preparing the data for clustering
clustering_data = merged_data[["Temperature Change", "CO₂ Concentration"]].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# applying K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # assuming 3 clusters for simplicity
clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)

# adding labels for periods with similar climate patterns
clustering_data['Label'] = clustering_data['Cluster'].map({
    0: 'Moderate Temp & CO₂',
    1: 'High Temp & CO₂',
    2: 'Low Temp & CO₂'
})

import plotly.express as px

fig_clusters = px.scatter(
    clustering_data,
    x="CO₂ Concentration",
    y="Temperature Change",
    color="Label",
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={
        "CO₂ Concentration": "CO₂ Concentration (ppm)",
        "Temperature Change": "Temperature Change (°C)",
        "Label": "Climate Pattern"
    },
    title="Clustering of Years Based on Climate Patterns"
)

fig_clusters.update_layout(
    template="plotly_white",
    legend_title="Climate Pattern"
)

fig_clusters.show()


# In[8]:


# setting up a simple predictive model using linear regression
from sklearn.linear_model import LinearRegression

# Preparing data
X = merged_data[["CO₂ Concentration"]].values  # CO₂ concentration as input
y = merged_data["Temperature Change"].values   # temperature change as target

model = LinearRegression()
model.fit(X, y)

# function to simulate "what-if" scenarios
def simulate_temperature_change(co2_percentage_change):
    # Calculate new CO2 concentrations
    current_mean_co2 = merged_data["CO₂ Concentration"].mean()
    new_co2 = current_mean_co2 * (1 + co2_percentage_change / 100)

    # predict temperature change
    predicted_temp = model.predict([[new_co2]])
    return predicted_temp[0]

# simulating scenarios
scenarios = {
    "Increase CO₂ by 10%": simulate_temperature_change(10),
    "Decrease CO₂ by 10%": simulate_temperature_change(-10),
    "Increase CO₂ by 20%": simulate_temperature_change(20),
    "Decrease CO₂ by 20%": simulate_temperature_change(-20),
}

scenarios


# In[ ]:




