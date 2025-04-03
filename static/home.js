document.getElementById("send").addEventListener("click", function() {
    let city = document.getElementById("city").value;
    let days = document.getElementById("days").value;
    
    fetch("http://127.0.0.1:5000/itinerary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ city: city, days: parseInt(days) })
    })
    .then(response => response.json())
    .then(data => {
        let outputDiv = document.getElementById("output");
        outputDiv.innerHTML = "";
        
        if (data.error) {
            outputDiv.innerHTML = `<p class='text-red-600 text-lg'>${data.error}</p>`;
            return;
        }
        
        let itineraryHtml = data.itinerary.map(day => `
            <div class='bg-white bg-opacity-80 p-6 rounded-lg shadow-md mb-6 w-full'>
                <h2 class='text-2xl font-bold text-gray-900 mb-3'>Day ${day.day}</h2>
                <p><strong class='text-blue-700'>Morning:</strong> <span class='text-lg font-bold text-gray-800'>${day.morning.place}</span> - ${day.morning.review}</p>
                <p><strong class='text-blue-700'>Noon:</strong> <span class='text-lg font-bold text-gray-800'>${day.noon.place}</span> - ${day.noon.review}</p>
                <p><strong class='text-blue-700'>Evening:</strong> <span class='text-lg font-bold text-gray-800'>${day.evening.place}</span> - ${day.evening.review}</p>
            </div>
        `).join('');
        
        if (data.recommended_hotel) {
            let hotel = data.recommended_hotel;
            itineraryHtml += `
                <div class='bg-green-100 p-6 rounded-lg shadow-md mt-6 w-full'>
                    <h2 class='text-2xl font-bold text-green-800 mb-3'>Recommended Hotel</h2>
                    <p><strong>Name:</strong> <span class='text-lg font-bold text-gray-900'>${hotel.name}</span></p>
                    <p><strong>Location:</strong> <span class='text-lg font-bold text-red-600'>${hotel.location}</span></p>
                    <p><strong>Price:</strong> ₹${hotel.price}</p>
                </div>
            `;
        }
        
        outputDiv.innerHTML = itineraryHtml;
        
        gsap.from("#output div", { opacity: 0, y: 20, duration: 1, stagger: 0.3 });
    })
    .catch(error => console.error("Error:", error));
});