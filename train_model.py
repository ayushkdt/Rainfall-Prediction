import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv(r'C:\Users\anilk\Rainfallprediction\Rainfall.csv')

# Print columns to verify
print(data.columns)

# Identify categorical columns and convert them to numeric
# For example, if 'rainfall' column has 'yes'/'no', we'll convert it.
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns (replace 'rainfall' with any other categorical column names)
data['rainfall'] = label_encoder.fit_transform(data['rainfall'])

# Define features and target
X = data[['day', 'pressure', 'maxtemp', 'temperature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'rainfall', 'sunshine', 'winddirection', 'windspeed']]
y = data['rainfall']  # Assuming 'rainfall' is the target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
