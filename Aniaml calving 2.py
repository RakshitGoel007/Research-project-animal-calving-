import mysql.connector
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
# 1. MySQL Database functions
def fetch_data_from_db1():
    try:
        # Connect to Database 1 (Source Database)
        db1_connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysqlgoel',
            database='db1_name'  # Ensure this database exists
        )
        cursor1 = db1_connection.cursor(dictionary=True)
        # Query to fetch data from Database 1
        cursor1.execute("SELECT * FROM source_table")  # Ensure this table exists
        print("Fetching data from Database 1...")  # Debugging
        # Fetch all the rows
        data_from_db1 = cursor1.fetchall()
        print(f"Fetched {len(data_from_db1)} rows from Database 1.")  # Debugging
        return data_from_db1
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    finally:
        if cursor1:
            cursor1.close()
        if db1_connection:
            db1_connection.close()
def insert_data_into_db2(data):
    try:
        # Connect to Database 2 (Destination Database)
        db2_connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysqlgoel',
            database='db2_name'  # Ensure this database exists
        )
        cursor2 = db2_connection.cursor()
        # Prepare an insert query (adjust the column names as needed)
        insert_query = """
        INSERT INTO destination_table (column1, column2, column3, predicted_activity)
        VALUES (%s, %s, %s, %s)
        """
        if data:
            print(f"Inserting {len(data)} rows into Database 2.")  # Debugging
            # Insert data into Database 2 with prediction
            for row in data:
                cursor2.execute(insert_query, (row['column1'], row['column2'], row['column3'], row['predicted_activity']))
            # Commit the transaction
            db2_connection.commit()
            print("Data inserted successfully.")
        else:
            print("No data to insert.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if cursor2:
            cursor2.close()
        if db2_connection:
            db2_connection.close()
# 2. ML Model training and prediction
def train_and_save_model():
    print("Training and saving the model...")  # Debugging
    # Load your data
    df = pd.read_excel('Calvingdata_p4.xlsx')  # Replace with actual data path
    print(f"Data loaded from Excel: {df.shape[0]} rows.")  # Debugging
    df_cleaned = df.dropna().reset_index(drop=True)
    
    # Feature engineering
    def calculate_tilt_angle(accel_x, accel_y, accel_z):
        return np.arctan2(np.sqrt(accel_x**2 + accel_y**2), accel_z) * (180.0 / np.pi)
    
    df_cleaned['tilt_angle'] = calculate_tilt_angle(df_cleaned['accel_x'], df_cleaned['accel_y'], df_cleaned['accel_z'])
    
    # Labeling activities
    def classify_activity(row):
        if 70 <= row['tilt_angle'] <= 90 and np.abs(row[['gyro_x', 'gyro_y', 'gyro_z']]).max() < 0.5:
            return 'Lying'
        elif row['tilt_angle'] < 30:
            return 'Standing'
        else:
            return 'Sitting'
    
    df_cleaned['label'] = df_cleaned.apply(classify_activity, axis=1)
    
    # Features and target variable
    X = df_cleaned[['accel_x', 'accel_z', 'tilt_angle']]
    y = df_cleaned['label']
    
    # Balancing the dataset using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.5, random_state=42)
    
    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Save the model
    dump(clf, 'activity_classifier_model.pkl')
    print("Model saved as 'activity_classifier_model.pkl'.")  # Debugging
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))  # Debugging
    print("Classification report:\n", classification_report(y_test, y_pred))  # Debugging
# 3. Predict activity based on incoming data (database rows)
def predict_activity(row):
    try:
        # Load the model
        model = load('activity_classifier_model.pkl')
        print(f"Model loaded: {model}")  # Debugging
        # Prepare features for prediction
        features = [row['accel_x'], row['accel_z'], row['tilt_angle']]  # Ensure these keys exist in row
        print(f"Prediction features: {features}")  # Debugging
        # Make the prediction
        prediction = model.predict([features])
        print(f"Predicted activity: {prediction[0]}")  # Debugging
        # Return predicted activity label
        return prediction[0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
# 4. Main logic
def main():
    # Step 1: Fetch data from Database 1
    data_from_db1 = fetch_data_from_db1()
    if not data_from_db1:
        print("No data fetched from Database 1. Exiting.")
        return
    
    # Step 2: Process the data and make predictions
    processed_data = []
    for row in data_from_db1:
        # Ensure 'accel_x', 'accel_y', 'gyro_x', etc., exist
        if 'accel_x' in row and 'accel_y' in row and 'accel_z' in row:
            # Calculate tilt_angle for each row
            row['tilt_angle'] = np.arctan2(np.sqrt(row['accel_x']**2 + row['accel_y']**2), row['accel_z']) * (180.0 / np.pi)
            # Predict the activity based on the model
            predicted_activity = predict_activity(row)
            if predicted_activity:
                row['predicted_activity'] = predicted_activity
            else:
                row['predicted_activity'] = 'Unknown'
        else:
            row['predicted_activity'] = 'Invalid Data'
        processed_data.append(row)
    
    # Step 3: Insert the processed data (with predictions) into Database 2
    insert_data_into_db2(processed_data)
if __name__ == "__main__":
    # Uncomment this if you want to train the model
    # train_and_save_model()
    
    # Run the main function to fetch, predict, and insert data
    main()
