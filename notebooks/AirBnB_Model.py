# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## AirBnB Optimal Price Predictor
# 
# This notebook downloads the data from remote used in this project and then cleans it. Ready to be used to build a model for predictions.
# 

# %%
# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from sklearn.pipeline import make_pipeline
import numpy as np


# %%
df = pd.read_csv('../data/raw/australia_visualisations_listings.csv.csv')

df


# %%
# Check out for any null values
df.isnull().sum()
# No null values


# %%
# Drop unnecessary columns and
# Rearrange columns in X, y format
df = df[['latitude', 'longitude', 'room_type',
       'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count',
       'availability_365', 'price']]


# %%
df.describe(include='all')


# %%
# Split in X and y

X = df[['latitude', 'longitude', 'room_type',
       'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count',
       'availability_365']]
y = df[['price']]

X.shape, y.shape


# %%
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15, shuffle=True)

# See the shape
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# %%
# TODO: Investigate why this is not working
"""
# OneHot Encode data
def ohe_transform(X_train, X_test):
       """
"""  OneHotEncoder transformer for X_train, X_test data categorical information
       :param X_train:
       :param X_test:
       :return: X_train_ohe, X_test_ohe"""
"""
       ohe = OrdinalEncoder()
       ohe.fit(X_train)
       X_train_ohe = X_train.transform(X_train)
       X_test_ohe = X_train.transform(X_test)

       return X_train_ohe, X_test_ohe

X_train_ohe, X_test_ohe = ohe_transform(X_train, X_test)


# prepare input data in quantitative data
def transform_categorical_data(X_train, X_test):
       oe = OrdinalEncoder()
       oe.fit(X_train)
       X_train_oe = oe.transform(X_train)
       X_test_oe = oe.transform(X_test)
       return X_train_oe, X_test_oe

# Run the function to have encoded X_train and X_test
X_train, X_test = transform_categorical_data(X_train, X_test)

"""


# %%
"""
def transform_categorical_data(X_train, X_test):
       oe = OrdinalEncoder()
       oe.fit(X_train)
       X_train_oe = oe.transform(X_train)
       X_test_oe = oe.transform(X_test)
       return X_train_oe, X_test_oe

# Run the function to have encoded X_train and X_test
X_train_oe, X_test_oe = transform_categorical_data(X_train, X_test)"""


# %%
# The code above didn't work therefore using pipeline to do OneHotEncoding

# Define basic pipeline
pipeline = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True)
)

#fit on train, score on val
pipeline.fit(X_train, y_train)
# print('Val accuracy', pipeline.score(X_val, y_val))

#before encoding
X_train.shape

#after encoding
encoder = pipeline.named_steps['onehotencoder']
X_train_ohe = encoder.transform(X_train)
#
X_train_ohe.head()


# %%
# Transform X_test using OHE
X_test_ohe = encoder.transform(X_test)


# %%
# Export DataFrame data to numpy values
X_train_np = X_train_ohe.values
X_test_np = X_test_ohe.values


# %%
# Code below cannot be used to save model due to the presence of custom model
"""
# Defining metric coeff_determination to get R-squared value for linear regression metric for target y
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Defining vanilla Feedforward neural network
model = Sequential()
model.add(Dense(10, input_dim=X_train_np.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='linear'))

# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy' ,'mean_absolute_error'])
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[coeff_determination])

model.summary()"""


# %%
# Defining vanilla Feedforward neural network
model = Sequential()
model.add(Dense(10, input_dim=X_train_np.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='linear'))

# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy' ,'mean_absolute_error'])
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy', 'mean_absolute_error'])

model.summary()


# %%
# Fit the model
model.fit(X_train_np, y_train, batch_size=128,epochs=5, verbose=1, validation_data=(X_test_ohe, y_test))

# %% [markdown]
# ### The losses seems high and coeff_determination is very low. Not a good model

# %%
predicted_prices = model.predict(X_test_ohe)
predicted_prices


# %%
# Actual
y_test


# %%
# Checkout a prediction from the model
test_prediction = {'latitude': -38.25482, 'longitude': 144.50328, 'room_type_Entire home/apt': 1,
       'room_type_Private room': 0, 'room_type_Shared room': 0,
       'room_type_Hotel room': 0, 'minimum_nights': 2, 'number_of_reviews': 3,
       'calculated_host_listings_count': 1, 'availability_365': 31}
test_prediction_df = pd.DataFrame([test_prediction])

model.predict(test_prediction_df)


# %%
# Save Model
model.save('../models/airbnbpredict.h5')


# %%
# Test by loading and predicting

# load model
test_model = load_model('../models/airbnbpredict.h5')

# Model Summary
test_model.summary()

# Checkout a prediction from the model
test_prediction = {'latitude': -38.25482, 'longitude': 144.50328, 'room_type_Entire home/apt': 1,
       'room_type_Private room': 0, 'room_type_Shared room': 0,
       'room_type_Hotel room': 0, 'minimum_nights': 2, 'number_of_reviews': 3,
       'calculated_host_listings_count': 1, 'availability_365': 31}
test_prediction_df = pd.DataFrame([test_prediction])

print(test_model.predict(test_prediction_df))


# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



