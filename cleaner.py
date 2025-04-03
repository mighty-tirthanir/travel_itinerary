import pandas as pd

# --- File Paths ---
input_path = './Review_db.csv'           # Input dataset path
output_path = './Cleaned_Review_db.csv'  # Output cleaned dataset path

# --- Load Dataset ---
print("📊 Loading dataset...")
df = pd.read_csv(input_path)

# --- Display basic info ---
print("\n✅ Initial Dataset Shape:", df.shape)
print("Columns:", df.columns)

# --- Column Names (Modify if needed) ---
city_col = 'City'       # City column name
place_col = 'Place'     # Place column name
rating_col = 'Rating'   # Rating column name
review_col = 'Review'   # Review column name

# --- Remove rows with missing values in key columns ---
df = df.dropna(subset=[city_col, place_col, rating_col, review_col])

# --- Convert Rating to Numeric for Sorting ---
df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')

# --- Remove Empty or Meaningless Reviews ---
df = df[df[review_col].str.strip() != '']  # Remove empty reviews

# --- Keep Only the Highest Rated Entry per Place ---
df = df.sort_values(by=[rating_col], ascending=False)  # Sort by highest rating
df = df.drop_duplicates(subset=[city_col, place_col], keep='first')  # Keep highest-rated entry per (City, Place)

# --- Save the Cleaned Dataset ---
df.to_csv(output_path, index=False)

# --- Summary ---
print("\n✅ Cleaning Complete!")
print(f"Cleaned dataset saved at: {output_path}")
print("Final Shape:", df.shape)
