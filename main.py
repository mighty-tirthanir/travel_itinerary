import pandas as pd
import pickle

# --- Load Datasets ---
def load_data():
    review_df = pd.read_csv('./Updated_Review_db3.csv', dtype=str, low_memory=False)
    hotel_df = pd.read_csv('./HotelData_cleaned.csv')  # Updated file
    restaurant_df = pd.read_csv('./restaurants_cleaned.csv')

    print("✅ Data Loaded Successfully!")

    return review_df, hotel_df, restaurant_df

# --- Load ML Model & Vectorizer ---
with open("location_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def rank_places_by_ml(city_reviews):
    """Ranks places using ML model predictions."""
    if city_reviews.empty or "Review" not in city_reviews.columns:
        return city_reviews  # Return as-is if no reviews
    
    X = vectorizer.transform(city_reviews["Review"])
    scores = model.predict(X)
    city_reviews["ML_Score"] = scores

    return city_reviews.sort_values(by="ML_Score", ascending=False)  # Sort by ML score

# --- Generate Itinerary ---
def generate_itinerary(city, days):
    review_df, hotel_df, restaurant_df = load_data()

    # Normalize city names
    city_lower = city.lower().strip()

    # Filter relevant data
    city_reviews = review_df[review_df["City"].str.contains(city_lower, case=False, na=False)]
    hotels = hotel_df[hotel_df["city"].str.contains(city_lower, case=False, na=False)]
    restaurants = restaurant_df[restaurant_df["City"].str.contains(city_lower, case=False, na=False)]

    # Rank places using ML model
    if not city_reviews.empty:
        city_reviews = rank_places_by_ml(city_reviews)

    # Initialize itinerary
    daily_itinerary = []

    # Plan for each day
    for day in range(1, days + 1):
        day_plan = {"Day": day}

        # Pick places (morning, noon, evening)
        places = city_reviews.iloc[(day - 1) * 3 : day * 3] if not city_reviews.empty else []
        places = places.to_dict("records") if not isinstance(places, list) else places

        if len(places) > 0:
            day_plan["Morning"] = places[0]
        if len(places) > 1:
            day_plan["Noon"] = places[1]
        if len(places) > 2:
            day_plan["Evening"] = places[2]

        # Pick restaurants (lunch, dinner) - Ensure it's not empty before accessing
        if not restaurants.empty:
            lunch_index = min(day - 1, len(restaurants) - 1)
            dinner_index = min(day, len(restaurants) - 1)

            lunch_restaurant = restaurants.iloc[lunch_index]
            dinner_restaurant = restaurants.iloc[dinner_index]

            day_plan["Lunch"] = {
                "Name": lunch_restaurant.get("Name", "Unknown"),
                "Location": lunch_restaurant.get("Location", "Unknown"),
                "Cuisine": lunch_restaurant.get("Cuisine", "Unknown"),
                "Price": lunch_restaurant.get("Cost", "N/A"),
                "Rating": lunch_restaurant.get("Rating", "N/A"),
            } if not lunch_restaurant.empty else None

            day_plan["Dinner"] = {
                "Name": dinner_restaurant.get("Name", "Unknown"),
                "Location": dinner_restaurant.get("Location", "Unknown"),
                "Cuisine": dinner_restaurant.get("Cuisine", "Unknown"),
                "Price": dinner_restaurant.get("Cost", "N/A"),
                "Rating": dinner_restaurant.get("Rating", "N/A"),
            } if not dinner_restaurant.empty else None

        daily_itinerary.append(day_plan)

    # Pick best hotel (if available)
    best_hotel = hotels.iloc[0] if not hotels.empty else None

    return daily_itinerary, best_hotel

# --- Display Itinerary ---
def display_itinerary(itinerary, best_hotel):
    print("\n✅ Travel Itinerary Recommendation")

    for day in itinerary:
        print(f"\n📅 Day {day['Day']}")
        
        if "Morning" in day:
            print(f"  🌞 Morning Visit: {day['Morning'].get('Place', 'Unknown')} (⭐ {day['Morning'].get('Rating', 'N/A')})")
            print(f"     - {day['Morning'].get('Review', 'No review available')}")

        if "Noon" in day:
            print(f"  🌆 Noon Visit: {day['Noon'].get('Place', 'Unknown')} (⭐ {day['Noon'].get('Rating', 'N/A')})")
            print(f"     - {day['Noon'].get('Review', 'No review available')}")

        if "Evening" in day:
            print(f"  🌙 Evening Visit: {day['Evening'].get('Place', 'Unknown')} (⭐ {day['Evening'].get('Rating', 'N/A')})")
            print(f"     - {day['Evening'].get('Review', 'No review available')}")

        if "Lunch" in day and day["Lunch"]:
            print(f"  🍽️ Lunch: {day['Lunch']['Name']} (⭐ {day['Lunch']['Rating']})")
            print(f"     - Cuisine: {day['Lunch']['Cuisine']}")
            print(f"     - Price: ₹{day['Lunch']['Price']}")
            print(f"     - Location: {day['Lunch']['Location']}")

        if "Dinner" in day and day["Dinner"]:
            print(f"  🍕 Dinner: {day['Dinner']['Name']} (⭐ {day['Dinner']['Rating']})")
            print(f"     - Cuisine: {day['Dinner']['Cuisine']}")
            print(f"     - Price: ₹{day['Dinner']['Price']}")
            print(f"     - Location: {day['Dinner']['Location']}")

    # Display hotel recommendation
    if best_hotel is not None:
        print("\n🏨 Recommended Hotel:")
        print(f"  - Name: {best_hotel.get('name', 'Unknown')}")
        print(f"  - Location: {best_hotel.get('address', 'Unknown')}")
        print(f"  - Price: ₹{best_hotel.get('price', 'N/A')}")
    else:
        print("\n❌ No hotels found for this city.")

# --- Main Execution ---
def main():
    print("🔍 Welcome to the Travel Itinerary Recommendation System! 🌍")

    # Get user input
    city = input("Enter the city: ").strip()
    days = int(input("Enter the number of days: ").strip())

    # Generate and display itinerary
    daily_itinerary, best_hotel = generate_itinerary(city, days)
    display_itinerary(daily_itinerary, best_hotel)

# Run program
if __name__ == "__main__":
    main()
