import pandas as pd
import pickle
import json

def load_data():
    review_df = pd.read_csv('./Updated_Review_db3.csv', dtype=str, low_memory=False)
    hotel_df = pd.read_csv('./HotelData_cleaned.csv')
    restaurant_df = pd.read_csv('./restaurants_cleaned.csv')
    print("✅ Data Loaded Successfully!")
    return review_df, hotel_df, restaurant_df

with open("location_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def rank_places_by_ml(city_reviews):
    if city_reviews.empty or "Review" not in city_reviews.columns:
        return city_reviews
    X = vectorizer.transform(city_reviews["Review"])
    scores = model.predict(X)
    city_reviews["ML_Score"] = scores
    return city_reviews.sort_values(by="ML_Score", ascending=False)

def generate_itinerary(city, days):
    review_df, hotel_df, restaurant_df = load_data()
    city_lower = city.lower().strip()
    
    city_reviews = review_df[review_df["City"].str.lower().str.strip() == city_lower]
    hotels = hotel_df[hotel_df["city"].str.lower().str.strip() == city_lower]
    restaurants = restaurant_df[restaurant_df["City"].str.lower().str.strip() == city_lower]

    # 🔴 If city doesn't exist in any dataset
    if city_reviews.empty and hotels.empty and restaurants.empty:
        print(f"❌ City '{city}' not found in the dataset. Please try a different city.")
        return None

    if not city_reviews.empty:
        city_reviews = rank_places_by_ml(city_reviews)
    
    itinerary = []
    for day in range(1, days + 1):
        day_plan = {"day": day}
        
        places = city_reviews.iloc[(day - 1) * 3 : day * 3] if not city_reviews.empty else []
        places = places.to_dict("records") if not isinstance(places, list) else places
        
        if len(places) > 0:
            day_plan["morning"] = {"place": places[0]["Place"], "rating": float(places[0]["Rating"]), "review": places[0]["Review"]}
        if len(places) > 1:
            day_plan["noon"] = {"place": places[1]["Place"], "rating": float(places[1]["Rating"]), "review": places[1]["Review"]}
        if len(places) > 2:
            day_plan["evening"] = {"place": places[2]["Place"], "rating": float(places[2]["Rating"]), "review": places[2]["Review"]}
        
        if not restaurants.empty:
            lunch_index = min(day - 1, len(restaurants) - 1)
            dinner_index = min(day, len(restaurants) - 1)
            
            lunch_restaurant = restaurants.iloc[lunch_index]
            dinner_restaurant = restaurants.iloc[dinner_index]
            
            day_plan["lunch"] = {
                "name": lunch_restaurant.get("Name", "Unknown"),
                "rating": float(lunch_restaurant.get("Rating", 0)),
                "cuisine": lunch_restaurant.get("Cuisine", "").split(", "),
                "price": int(lunch_restaurant.get("Cost", 0)),
                "location": lunch_restaurant.get("Location", "Unknown")
            }
            
            day_plan["dinner"] = {
                "name": dinner_restaurant.get("Name", "Unknown"),
                "rating": float(dinner_restaurant.get("Rating", 0)),
                "cuisine": dinner_restaurant.get("Cuisine", "").split(", "),
                "price": int(dinner_restaurant.get("Cost", 0)),
                "location": dinner_restaurant.get("Location", "Unknown")
            }
        
        itinerary.append(day_plan)
    
    best_hotel = hotels.iloc[0] if not hotels.empty else None
    
    return {
        "city": city,
        "days": days,
        "itinerary": itinerary,
        "recommended_hotel": {
            "name": best_hotel.get("name", "Unknown"),
            "location": best_hotel.get("address", "Unknown"),
            "price": best_hotel.get("price", "N/A")
        } if best_hotel is not None else None
    }

def main():
    print("🔍 Welcome to the Travel Itinerary Recommendation System! 🌍")
    city = input("Enter the city: ").strip()
    days = int(input("Enter the number of days: ").strip())
    
    itinerary_json = generate_itinerary(city, days)
    
    if itinerary_json:
        print(json.dumps(itinerary_json, indent=4))

if __name__ == "_main_":
    main()