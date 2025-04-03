export async function fetchOurModelItinerary(city, days) {
    try {
        let response = await fetch("http://127.0.0.1:5000/itinerary", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ city, days: parseInt(days) })
        });
        return await response.json();
    } catch (error) {
        console.error("Error fetching itinerary from Our Model:", error);
        return null;
    }
}