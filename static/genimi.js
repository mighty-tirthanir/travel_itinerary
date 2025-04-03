export async function fetchGeminiItinerary(city, days) {
    const apiKey = "YOUR_GEMINI_API_KEY";
    const prompt = `
        Generate a ${days}-day travel itinerary for ${city}.
        Each day should have:
        - Morning activity (location & short review)
        - Noon activity (location & short review)
        - Evening activity (location & short review)
        - Recommended lunch and dinner spots with name, location, and approximate cost.
        - A recommended hotel with name, location, and price.
        Return in JSON format.
    `;

    try {
        let response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateText?key=${apiKey}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt })
        });

        let data = await response.json();
        return JSON.parse(data.candidates[0].content); // Extract JSON itinerary from response
    } catch (error) {
        console.error("Error fetching itinerary from Gemini:", error);
        return null;
    }
}