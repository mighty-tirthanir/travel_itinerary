import pandas as pd
import numpy as np

# --- Load Dataset ---
df = pd.read_csv('Final_Updated_Review_db3.csv', dtype=str, low_memory=False)

# Drop missing values
df = df.dropna(subset=['City', 'Place', 'Review', 'Rating'])

# Convert Rating to numeric
df['Rating'] = df['Rating'].astype(float)

# 🔹 **Randomly Select 3000 Unique Places & Assign Rating 10**
unique_places = df['Place'].unique()  # Get unique places
selected_places = np.random.choice(unique_places, 3000, replace=False)  # Randomly pick 3000

# Update ratings for the selected places
df.loc[df['Place'].isin(selected_places), 'Rating'] = 10.0

# --- Save Updated Dataset ---
df.to_csv('Updated_Review_db3.csv', index=False)

print("\n✅ Successfully assigned rating 10 to 3,000 random places & saved the dataset!")
