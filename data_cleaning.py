import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# **YouTube Data Cleaning**
# Load all 10 YouTube datasets dynamically from the 'data' folder
# Load all 10 YouTube datasets dynamically from the 'data' folder
files = os.listdir('data')  # List all files in the 'data' folder
youtube_dfs = []

for file in files:
    if file.endswith('.csv'):  # Ensure it's a CSV file
        youtube_dfs.append(pd.read_csv(f'data/{file}', encoding='ISO-8859-1'))  # Try using 'ISO-8859-1' encoding

# Combine all YouTube data into one DataFrame
youtube_df = pd.concat(youtube_dfs, ignore_index=True)


# Check for missing data in YouTube dataset
print("Missing data in YouTube dataset:")
print(youtube_df.isnull().sum())

# Handle missing data - Fill missing values with median (for numerical) and mode (for categorical)
youtube_df['views'].fillna(youtube_df['views'].median(), inplace=True)
youtube_df['likes'].fillna(youtube_df['likes'].median(), inplace=True)
youtube_df['dislikes'].fillna(youtube_df['dislikes'].median(), inplace=True)
youtube_df['category_id'].fillna(youtube_df['category_id'].mode()[0], inplace=True)

# Check for duplicates
print("\nDuplicate rows in YouTube dataset:")
print(youtube_df.duplicated().sum())

# Remove duplicate rows
youtube_df.drop_duplicates(inplace=True)

# Standardization - Clean text columns like 'category' and 'tags'
# Convert category_id to string and apply string operations
youtube_df['category_id'] = youtube_df['category_id'].astype(str)
youtube_df['category_id'] = youtube_df['category_id'].str.strip().str.lower()


# Outlier Detection - Remove outliers in 'views' using IQR method
Q1 = youtube_df['views'].quantile(0.25)
Q3 = youtube_df['views'].quantile(0.75)
IQR = Q3 - Q1
youtube_df = youtube_df[(youtube_df['views'] >= (Q1 - 1.5 * IQR)) & (youtube_df['views'] <= (Q3 + 1.5 * IQR))]

# Save cleaned YouTube dataset
youtube_df.to_csv('data/cleaned_youtube.csv', index=False)

# **Airbnb Data Cleaning**
# Load Airbnb dataset
airbnb_df = pd.read_csv('data/airbnb.csv')

# Check for missing data in Airbnb dataset
print("Missing data in Airbnb dataset:")
print(airbnb_df.isnull().sum())

# Handle missing data - Drop rows with missing values for simplicity
airbnb_df.dropna(inplace=True)

# Check for duplicates
print("\nDuplicate rows in Airbnb dataset:")
print(airbnb_df.duplicated().sum())

# Remove duplicate rows
airbnb_df.drop_duplicates(inplace=True)

# Standardization - Make 'room_type' lowercase and strip any extra spaces
airbnb_df['room_type'] = airbnb_df['room_type'].str.lower().str.strip()

# Outlier Detection - Price outliers using IQR method
Q1 = airbnb_df['price'].quantile(0.25)
Q3 = airbnb_df['price'].quantile(0.75)
IQR = Q3 - Q1
airbnb_df = airbnb_df[(airbnb_df['price'] >= (Q1 - 1.5 * IQR)) & (airbnb_df['price'] <= (Q3 + 1.5 * IQR))]

# Save cleaned Airbnb dataset
airbnb_df.to_csv('data/cleaned_airbnb.csv', index=False)

# **Visualizations (Optional)**

# Visualize distribution of price for Airbnb
sns.histplot(airbnb_df['price'], kde=True)
plt.title('Price Distribution for Airbnb Listings')
plt.show()

# Visualize distribution of views for YouTube
sns.histplot(youtube_df['views'], kde=True)
plt.title('Views Distribution for YouTube Videos')
plt.show()

# Boxplot for detecting outliers in Airbnb price
sns.boxplot(x=airbnb_df['price'])
plt.title('Price Outliers for Airbnb Listings')
plt.show()

# Boxplot for detecting outliers in YouTube views
sns.boxplot(x=youtube_df['views'])
plt.title('Views Outliers for YouTube Videos')
plt.show()

# **Conclusion**
print("Data cleaning complete for both Airbnb and YouTube datasets.")
