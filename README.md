# Categorical-Analysis-Correlation-Matrix
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset
data = pd.read_csv("used_cars.csv")

# Initial data inspection
print("Shape of the dataset:", data.shape)
print(data.head())
print(data.info())

# Handling missing values
print("Missing values in each column:")
print(data.isnull().sum())
data['Mileage'].fillna(value=np.mean(data['Mileage']), inplace=True)

# Feature Engineering
data['Car_Age'] = 2025 - data['Year']  # Assuming the current year is 2025

# Univariate Analysis
numerical_cols = data.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    data[col].hist(grid=False, color='blue', bins=30)
    plt.title(f'Histogram of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col], color='orange')
    plt.title(f'Boxplot of {col}')
    plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Kilometers_Driven', y='Price', data=data, color='purple')
plt.title('Kilometers Driven vs Price')
plt.xlabel('Kilometers Driven')
plt.ylabel('Price')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 7))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=1)
plt.title('Correlation Matrix')
plt.show()

# Categorical Analysis
plt.figure(figsize=(12, 6))
sns.countplot(x='Fuel_Type', data=data, palette='Set2')
plt.title('Count of Cars by Fuel Type')
plt.show()

# Violin Plot for Price vs Car Age
plt.figure(figsize=(10, 6))
sns.violinplot(x='Car_Age', y='Price', data=data, palette='muted')
plt.title('Price Distribution by Car Age')
plt.xlabel('Car Age')
plt.ylabel('Price')
plt.show()

# Final Insights
print("Summary of findings:")
# Add your insights based on the analysis here
