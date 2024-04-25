import pandas as pd
import numpy as np

# Define doctor names, hospital names, and city names
doctor_names = ["Dr. Alice Smith", "Dr. Bob Johnson", "Dr. Charlie Brown", "Dr. Olivia Jones", "Dr. Michael Lee"]
hospital_names = ["AIIMS", "Central City Medical Center", "Sunshine Valley Clinic", "Riverside Community Hospital", "Mountain View Medical Center"]
city_names = ["Delhi", "Pune", "Mumbai", "Hyderabad", "Banglore"]

# Generate synthetic patient data
num_patients = 1000
patients = pd.DataFrame({
  'Patient_ID': range(1, num_patients + 1),
  'Age': np.random.randint(18, 80, size=num_patients),
  'Gender': np.random.choice(['Male', 'Female'], size=num_patients),
  'History_of_Diseases': np.random.choice(['Yes', 'No'], size=num_patients)
})

# Select a random doctor name
random_doctor_index = np.random.randint(0, len(doctor_names))
recommended_doctor_name = doctor_names[random_doctor_index]

# Select a random hospital name and city
random_hospital_index = np.random.randint(0, len(hospital_names))
recommended_hospital_name = hospital_names[random_hospital_index]
recommended_city = city_names[random_hospital_index]  # Match city with hospital

# Simulate disease detection in a patient
patient_index = np.random.randint(0, num_patients)
disease_patient = patients.loc[patient_index, :]
disease_detected = np.random.choice(['Heart Disease', 'Tuberclosis', 'Tuberclosis', 'Skin Allergy'])


# Print results
print("Patient Details:")
print(disease_patient)
# print("\nDisease Detected:", disease_detected)
print("\nRecommended Doctor:")
print(recommended_doctor_name)
print("\nRecommended Hospital:")
print(recommended_hospital_name + ", " + recommended_city)
