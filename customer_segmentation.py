import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Create a Synthetic Dataset
num_customers = 1000

# Generate random data for customers
customer_id = np.arange(1, num_customers + 1)
age = np.random.randint(18, 70, size=num_customers)
gender = np.random.choice(['Male', 'Female'], size=num_customers)
location = np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_customers)
purchase_frequency = np.random.poisson(10, num_customers)
average_purchase_value = np.random.normal(50, 15, num_customers).clip(10, 200)
total_spend = purchase_frequency * average_purchase_value
customer_lifetime_value = total_spend * np.random.uniform(1, 3, num_customers)

# Create a DataFrame
df_customers = pd.DataFrame({
    'CustomerID': customer_id,
    'Age': age,
    'Gender': gender,
    'Location': location,
    'PurchaseFrequency': purchase_frequency,
    'AveragePurchaseValue': average_purchase_value,
    'TotalSpend': total_spend,
    'CustomerLifetimeValue': customer_lifetime_value
})

# Step 2: Perform Clustering (K-Means)
X = df_customers[['PurchaseFrequency', 'AveragePurchaseValue', 'CustomerLifetimeValue']]

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, we choose an optimal number of clusters
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df_customers['Cluster'] = kmeans.fit_predict(X)

# Step 3: Visualize the Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PurchaseFrequency', y='CustomerLifetimeValue', hue='Cluster', data=df_customers, palette='Set1')
plt.title('Customer Segments based on Purchase Frequency and Customer Lifetime Value')
plt.xlabel('Purchase Frequency')
plt.ylabel('Customer Lifetime Value')
plt.legend()
plt.show()

df_customers.head()  # Display the first few rows of the dataset to check the clusters and data
