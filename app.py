from flask import Flask, request, jsonify, render_template
from main import generate_itinerary

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/itinerary', methods=['POST'])
def get_itinerary():
    data = request.get_json()
    
    if not data or 'city' not in data or 'days' not in data:
        return jsonify({"error": "Missing city or days parameter"}), 400
    
    city = data['city']
    days = data['days']

    if not isinstance(days, int) or days <= 0:
        return jsonify({"error": "Days must be a positive integer"}), 400

    itinerary = generate_itinerary(city, days)
    print(itinerary)
    return jsonify(itinerary)

if __name__ == '__main__':
    app.run(debug=True)
