################################################################################
##########=           (Setup Code from Previous NBs)           =################
################################################################################
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go
# Load data

import requests

def download_with_requests(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {url} successfully to {output_path}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Example usage:
download_with_requests("https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Planet%20Hunters/exoTrain.csv", "exoTrain.csv")
download_with_requests("https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Planet%20Hunters/exoTest.csv", "exoTest.csv")

df_train = pd.read_csv('exoTrain.csv')
df_train['LABEL'] = df_train['LABEL'] - 1

# Load the CNN model
model = tf.keras.models.load_model('./cnn.keras')

def preprocess_data(df):
    """Apply preprocessing steps to the dataframe."""
    X = df.drop('LABEL', axis=1).values
    y = df['LABEL'].values
    # Fourier transform
    X = np.abs(np.fft.fft(X, axis=1))
    # Savitzky-Golay filter
    X = savgol_filter(X, 21, 4, deriv=0, axis=1)
    # Normalize
    X = normalize(X)
    # Robust scaling
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    # SMOTE
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
    # Expand dimensions for CNN input
    X_cnn = np.expand_dims(X, axis=2)
    return X, X_cnn, y

X_train, X_train_cnn, y_train = preprocess_data(df_train)

def predict(index):
    """Make a prediction using the Keras model."""
    tensor = X_train[index].reshape(1, -1, 1)
    output = model.predict(tensor)
    return output.flatten()[0]

################################################################################
##########=                   (Light Curves)                   =################
################################################################################
st.title('Exoplanet Light Curve Visualization with CNN Predictions')

# Slider for selecting the index of the light curve
index = st.slider("Select Index for Light Curve", min_value=0, max_value=len(X_train)-1, value=12, step=1)

# Display CNN prediction results
prediction = predict(index)
st.write(f"Prediction (probability of being an exoplanet): {prediction:.4f}")
t_0 = st.slider("Start of Period (t_0)", min_value=0, max_value=3197, value=430, step=1)
period = st.slider("Length of Period", min_value=0, max_value=3197, value=1184, step=1)

# Create Plotly graph for the full light curve with a rectangle highlighting the period
fig = go.Figure()
fig.add_trace(go.Scatter(y=X_train[index].flatten(), mode='lines', name='Light Curve'))
fig.add_shape(type="rect",
              x0=t_0,
              y0=min(X_train[index])-5,
              x1=t_0+period,
              y1=max(X_train[index])+5,
              line=dict(color="Red"),
              fillcolor="LightPink",
              opacity=0.5)
fig.update_layout(title="Box Covering One Period of Exoplanet Transit",
                  xaxis_title="Observation Point",
                  yaxis_title="Normalized Flux",
                  showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# Create Plotly graph for just the selected period
fig_period = go.Figure()
fig_period.add_trace(go.Scatter(y=X_train[index, t_0: t_0+period].flatten(), mode='lines', name='Selected Period'))
fig_period.update_layout(title="Plot of Just One Period",
                         xaxis_title="Observation Point",
                         yaxis_title="Normalized Flux",
                         showlegend=True)
st.plotly_chart(fig_period, use_container_width=True)

################################################################################
##########=                   (Confusion Matrix)               =################
################################################################################
st.title('Exoplanet Light Curve Visualization with CNN Predictions')


# Display CNN prediction results
prediction = predict(index)
st.write(f"Prediction (probability of being an exoplanet): {prediction:.4f}")

# Compute and plot confusion matrix
cnn_pred_labels = (model.predict(X_train).flatten() > 0.5).astype(int)
cm_cnn = confusion_matrix(y_train, cnn_pred_labels)

# Confusion Matrix for CNN
fig_cm_cnn = ff.create_annotated_heatmap(z=cm_cnn, x=['Pred 0', 'Pred 1'], y=['True 0', 'True 1'],
                                         colorscale='Viridis', showscale=True)
fig_cm_cnn.update_layout(title='Confusion Matrix for CNN for Entire Dataset')
st.plotly_chart(fig_cm_cnn, use_container_width=True)
################################################################################
##########=                   YOUR CODE BELOW                  =################
################################################################################
show_full_light_curve = st.checkbox("Show Full Light Curve", value=True)
if show_full_light_curve:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=X_train[index], mode='lines', name='Light Curve'))
    fig.add_shape(type="rect",
                  x0=t_0, y0=min(X_train[index])-5,
                  x1=t_0+period, y1=max(X_train[index])+5,
                  line=dict(color="Red"),
                  fillcolor="LightPink", opacity=0.5)
    fig.update_layout(title="Box Covering One Period of Exoplanet Transit",
                      xaxis_title="Observation Point",
                      yaxis_title="Normalized Flux",
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
