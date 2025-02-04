from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

# Use non-GUI backend for Matplotlib to prevent Tkinter issues
import matplotlib
matplotlib.use('Agg')

# Load Data
vibrational_data = pd.read_csv('Vibrational_Data.csv')
molecular_data = pd.read_excel('Molecular_Coordinates.xlsx')

# Data Preprocessing (Same as before)
molecular_agg = molecular_data.groupby('Molecule').agg({
    'X': 'mean', 'Y': 'mean', 'Z': 'mean', 'Number of Bonds': 'sum',
    'Dipole Moment': 'first', 'Point Group': 'first'
}).reset_index()

vibrational_agg = vibrational_data.groupby('Molecule').agg({
    'Frequency (cm-1)': list, 'IR Intensity': list
}).reset_index()

merged_data = pd.merge(molecular_agg, vibrational_agg, on='Molecule', how='inner')
label_encoder = LabelEncoder()
merged_data['Point Group Encoded'] = label_encoder.fit_transform(merged_data['Point Group'])

X = merged_data[['X', 'Y', 'Z', 'Number of Bonds', 'Dipole Moment', 'Point Group Encoded']]
target_length = 100  # Consistent output

y_frequencies = np.array([np.pad(freq, (0, max(0, target_length - len(freq))), 'constant')[:target_length] 
                          for freq in merged_data['Frequency (cm-1)']])
y_intensities = np.array([np.pad(intensity, (0, max(0, target_length - len(intensity))), 'constant')[:target_length] 
                          for intensity in merged_data['IR Intensity']])

# Train Models
frequency_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
intensity_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
frequency_model.fit(X, y_frequencies)
intensity_model.fit(X, y_intensities)

def gaussian_curve(x, center, intensity, sigma=10):
    return intensity * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def predict_and_plot(molecule_name):
    if molecule_name not in merged_data['Molecule'].values:
        return None
    
    molecule_index = merged_data[merged_data['Molecule'] == molecule_name].index[0]
    X_molecule = pd.DataFrame([X.iloc[molecule_index]], columns=X.columns)
    predicted_frequencies = frequency_model.predict(X_molecule)[0]
    predicted_intensities = intensity_model.predict(X_molecule)[0]
    
    actual_frequencies = np.array(merged_data.loc[molecule_index, 'Frequency (cm-1)'])
    actual_intensities = np.array(merged_data.loc[molecule_index, 'IR Intensity'])
    
    max_actual_intensity = actual_intensities.max()
    scaled_predicted_intensities = (predicted_intensities / predicted_intensities.max()) * max_actual_intensity
    
    actual_frequencies = np.pad(actual_frequencies, (0, target_length - len(actual_frequencies)), 'constant')
    actual_intensities = np.pad(actual_intensities, (0, target_length - len(actual_intensities)), 'constant')
    
    x = np.linspace(400, 4000, 500)
    actual_curve = np.sum([gaussian_curve(x, freq, inten) for freq, inten in zip(actual_frequencies, actual_intensities)], axis=0)
    predicted_curve = np.sum([gaussian_curve(x, freq, inten) for freq, inten in zip(predicted_frequencies, scaled_predicted_intensities)], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, actual_curve, label="Actual Intensity", linestyle="-", linewidth=2)
    plt.plot(x, predicted_curve, label="Scaled Predicted Intensity", linestyle="--", linewidth=2)
    plt.title(f"IR Spectrum for {molecule_name}")
    plt.xlabel("Wave Number (cm⁻¹)")
    plt.ylabel("IR Intensity")
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=50)  # Reduce DPI for performance
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

app = Flask(__name__)

# List of predefined test molecules
test_molecules = ["H2O", "CH3CH2OH", "NH3", "CH3CH2COOH", "C4H3I", "C6H5OH", "CH3CN", "C2H3NO", "CHCl3", "CO2"]

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    if request.method == 'POST':
        molecule_name = request.form['molecule_name']
        plot_url = predict_and_plot(molecule_name)
    
    return render_template('index.html', plot_url=plot_url, test_molecules=test_molecules)

if __name__ == '__main__':
    app.run(debug=True)