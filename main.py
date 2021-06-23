"""Instantiates the Airbnb application."""
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, redirect, request
from tensorflow.keras.models import load_model
from pathlib import Path



BASE_DIR = Path(__file__).parent.parent




app = Flask(__name__)
# Import model


model = load_model(filepath="airbnbpredict_all_data.h5")
# Define the column names for importing the user data
cols = ['latitude', 'longitude', 'neighbourhood', 'room_type',
        'minimum_nights', 'number_of_reviews',
        'calculated_host_listings_count', 'availability_365']


@app.route('/')
async def index():
    """Homepage view."""
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Get the information from the home.html and returns the predicted value."""
    # Collect the user data and change to array and then to DataFrame
    user_data = [float(x) for x in request.form.values()]
    user_data_np = np.array(user_data)
    user_data_df = pd.DataFrame([user_data_np], columns=[cols])
    # Use model to make prediction
    prediction = model.predict(user_data_df)

    if prediction < 0:
        return render_template('home.html',
                               pred='Your set parameters are out of range something trained model has not seen!')

    elif prediction > 2000000:
        return render_template('home.html',
                               pred='Your set parameters are out of range something! Please revisit them.')
    else:
        return render_template('home.html',
                               pred='Your property should be listed at {} in the '
                                    'local currency of the geo '
                                    'location.'.format(int(prediction[0][0])))


if __name__ == '__main__':
    app.run(debug=True)
