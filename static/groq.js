export async function fetchGroqItinerary(city, days) {
    const apiKey = "YOUR_GROQ_API_KEY";
    const prompt = `
        Create a ${days}-day travel itinerary for ${city}.
        Each day should include:
        - Morning activity with location & short review
        - Noon activity with location & short review
        - Evening activity with location & short review
        - Recommended places for lunch and dinner (name, location, price)
        - Suggested hotel (name, location, price)
        Return structured JSON format like:
        {
            "itinerary": [
                { "day": 1, "morning": {...}, "noon": {...}, "evening": {...}, "lunch": {...}, "dinner": {...} }
            ],
            "recommended_hotel": {...}
        }
    `;

    try {
        let response = await fetch("https://api.groq.com/v1/completions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${apiKey}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ model: "mixtral", prompt, max_tokens: 1000 })
        });

        let data = await response.json();
        return JSON.parse(data.choices[0].text); // Extract JSON itinerary from response
    } catch (error) {
        console.error("Error fetching itinerary from Groq:", error);
        return null;
    }
}