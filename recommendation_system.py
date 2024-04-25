import pandas as pd
import numpy as np

# Generate synthetic patient data
num_patients = 1000
patients = pd.DataFrame({
    'Patient_ID': range(1, num_patients + 1),
    'Age': np.random.randint(18, 80, size=num_patients),
    'Gender': np.random.choice(['Male', 'Female'], size=num_patients),
    'History_of_Diseases': np.random.choice(['Yes', 'No'], size=num_patients)
})

# Generate synthetic doctor data
num_doctors = 50
doctors = pd.DataFrame({
    'Doctor_ID': range(1, num_doctors + 1),
    'Specialization': np.random.choice(['Cardiologist', 'Orthopedist', 'Neurologist', 'Dermatologist'], size=num_doctors),
    'Rating': np.random.randint(1, 6, size=num_doctors)  # Assuming ratings from 1 to 5
})

# Generate synthetic hospital data
num_hospitals = 20
hospitals = pd.DataFrame({
    'Hospital_ID': range(1, num_hospitals + 1),
    'Location': ['City ' + str(i) for i in range(1, num_hospitals + 1)],
    'Rating': np.random.randint(1, 6, size=num_hospitals)  # Assuming ratings from 1 to 5
})

# Simulate disease detection in a patient
patient_index = np.random.randint(0, num_patients)
disease_patient = patients.loc[patient_index, :]
disease_detected = np.random.choice(['Heart Disease', 'Bone Fracture', 'Migraine', 'Skin Allergy'])

# Recommend doctor and hospital based on ratings
recommended_doctor = doctors[doctors['Specialization'] == 'Cardiologist'].sort_values(by='Rating', ascending=False).iloc[0]
recommended_hospital = hospitals.sort_values(by='Rating', ascending=False).iloc[0]

print("Patient Details:")
print(disease_patient)
print("\nDisease Detected:", disease_detected)
print("\nRecommended Doctor:")
print(recommended_doctor)
print("\nRecommended Hospital:")
print(recommended_hospital)
