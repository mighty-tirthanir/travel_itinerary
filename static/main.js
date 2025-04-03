import { fetchOurModelItinerary } from './ourModel.js';
import { fetchGeminiItinerary } from './geminiModel.js';
import { fetchGroqItinerary } from './groqModel.js';

document.getElementById("send").addEventListener("click", async function() {
    let city = document.getElementById("city").value;
    let days = document.getElementById("days").value;
    let selectedModel = document.getElementById("model").value;
    
    let data = null;
    if (selectedModel === "ourModel") {
        data = await fetchOurModelItinerary(city, days);
    } else if (selectedModel === "gemini") {
        data = await fetchGeminiItinerary(city, days);
    } else if (selectedModel === "groq") {
        data = await fetchGroqItinerary(city, days);
    }

    displayItinerary(data);
});

function displayItinerary(data) {
    let outputDiv = document.getElementById("output");
    outputDiv.innerHTML = "";

    if (!data || !data.itinerary || !Array.isArray(data.itinerary)) {
        outputDiv.innerHTML = `<p class='text-red-600 text-lg text-center font-semibold'>No itinerary available.</p>`;
        return;
    }

    let itineraryHtml = data.itinerary.map(day => `
        <div class='itinerary-card'>
            <h2 class='text-3xl font-bold text-gray-900 mb-3'>Day ${day.day}</h2>
            ${day.morning ? `<p class='text-lg'><strong class='text-blue-700'>Morning:</strong> ${day.morning.place} - ${day.morning.review}</p>` : ''}
            ${day.noon ? `<p class='text-lg'><strong class='text-blue-700'>Noon:</strong> ${day.noon.place} - ${day.noon.review}</p>` : ''}
            ${day.evening ? `<p class='text-lg'><strong class='text-blue-700'>Evening:</strong> ${day.evening.place} - ${day.evening.review}</p>` : ''}
            ${day.lunch ? `<p class='text-lg text-green-700'><strong>Lunch:</strong> ${day.lunch.name} - ${day.lunch.location} (₹${day.lunch.price})</p>` : ''}
            ${day.dinner ? `<p class='text-lg text-red-700'><strong>Dinner:</strong> ${day.dinner.name} - ${day.dinner.location} (₹${day.dinner.price})</p>` : ''}
        </div>
    `).join('');

    if (data.recommended_hotel) {
        let hotel = data.recommended_hotel;
        itineraryHtml += `
            <div class='itinerary-card bg-green-100'>
                <h2 class='text-3xl font-bold text-green-800 mb-3'>Recommended Hotel</h2>
                <p class='text-lg'><strong>Name:</strong> ${hotel.name}</p>
                <p class='text-lg'><strong>Location:</strong> ${hotel.location}</p>
                <p class='text-lg'><strong>Price:</strong> ₹${hotel.price}</p>
            </div>
        `;
    }

    outputDiv.innerHTML = itineraryHtml;
    gsap.to(".glass", { width: "90%" });
    gsap.from("#output div", { opacity: 0, y: 20, duration: 1, stagger: 0.3, delay: 0.4 });
}