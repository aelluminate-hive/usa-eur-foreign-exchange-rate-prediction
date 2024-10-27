import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set figure size and resolution for all plots
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['figure.dpi'] = 300

# Get the directory of the current script and construct the path to the CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))
daily_file = os.path.join(current_dir, 'daily.csv')

# Read the CSV file, selecting only Date, Country, and Exchange rate columns
data = pd.read_csv(daily_file, usecols=['Date', 'Country', 'Exchange rate'])
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Create a directory to store the graphs
graphs_dir = os.path.join(current_dir, 'graphs')
os.makedirs(graphs_dir, exist_ok=True)

# Remove old PNG files
for file in os.listdir(graphs_dir):
    if file.endswith(".png"):
        os.remove(os.path.join(graphs_dir, file))

print("Old PNG files have been removed.")

def add_plot_details(title, xlabel, ylabel):
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)

# 1. Line plot of exchange rates over time for each country
plt.figure(figsize=(20, 10))
for country in data['Country'].unique():
    country_data = data[data['Country'] == country]
    plt.plot(country_data['Date'], country_data['Exchange rate'], label=country, linewidth=1)
add_plot_details('Exchange Rates Over Time by Country', 'Date', 'Exchange Rate')
plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'exchange_rates_over_time.png'), bbox_inches='tight')
plt.close()

# 2. Box plot of exchange rates by country
plt.figure(figsize=(20, 10))
sns.boxplot(x='Country', y='Exchange rate', data=data)
add_plot_details('Exchange Rate Distribution by Country', 'Country', 'Exchange Rate')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'exchange_rates_boxplot_by_country.png'), bbox_inches='tight')
plt.close()

# 3. Line plot of exchange rates over time for all countries in one frame
plt.figure(figsize=(20, 10))

countries = data['Country'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))

for country, color in zip(countries, colors):
    country_data = data[data['Country'] == country]
    plt.plot(country_data['Date'], country_data['Exchange rate'], 
             label=country, linewidth=1, color=color, alpha=0.7)

plt.title('Exchange Rates Over Time by Country', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'exchange_rates_by_country.png'), bbox_inches='tight')
plt.close()

# 4. Heatmap of average exchange rates by year and country
yearly_avg = data.groupby([data['Date'].dt.year, 'Country'])['Exchange rate'].mean().unstack()
plt.figure(figsize=(20, 12))
sns.heatmap(yearly_avg, cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Exchange Rate'})
add_plot_details('Average Exchange Rates by Year and Country', 'Country', 'Year')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'exchange_rates_heatmap.png'), bbox_inches='tight')
plt.close()

# 5. Bar plot of average exchange rate by country
avg_by_country = data.groupby('Country')['Exchange rate'].mean().sort_values(ascending=False)
plt.figure(figsize=(20, 10))
avg_by_country.plot(kind='bar')
add_plot_details('Average Exchange Rate by Country', 'Country', 'Average Exchange Rate')
plt.xticks(rotation=90)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'average_exchange_rate_by_country.png'), bbox_inches='tight')
plt.close()

# 6. Line plot of exchange rate volatility over time
data['Year'] = data['Date'].dt.year
volatility = data.groupby(['Year', 'Country'])['Exchange rate'].std().unstack()
plt.figure(figsize=(20, 10))
volatility.plot(linewidth=2, marker='o')
add_plot_details('Exchange Rate Volatility Over Time by Country', 'Year', 'Exchange Rate Standard Deviation')
plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'exchange_rate_volatility.png'), bbox_inches='tight')
plt.close()

print("All new graphs have been generated and saved in the 'graphs' directory.")
