"""Instantiates the Airbnb application."""
from flask import Flask, render_template, redirect, request
from flask.helpers import url_for

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
async def index():
    """Homepage view."""

    if request.method == 'POST':
        latitude = request.form['latitude'],
        longitude = request.form['longitude'],
        room_type = request.form['room_type'],
        min_nights = request.form['min_nights'],
        number_of_reviews = request.form['number_of_reviews'],
        host_listing_count = request.form['host_listing_count'],
        availability_365 = request.form['availability_365']

        return redirect(
            url_for('predict')
        )

    else:
        return render_template('home.html')


@app.route('/predict')
def predict():
    """Runs the user values through the price optimization model."""
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
