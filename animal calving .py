import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Create database, table, and load CSV data into db1
def setup_database_and_load_data():
    try:
        # Connect to MySQL (without specifying a database initially)
        db_connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysqlgoel'
        )
        cursor = db_connection.cursor()

        # Create database and table if they do not exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS db1_name;")
        cursor.execute("USE db1_name;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Values_from_arduino(
                accel_x FLOAT,
                accel_y FLOAT,
                accel_z FLOAT,
                gyro_x FLOAT,
                gyro_y FLOAT,
                gyro_z FLOAT
            );
        """)

        # Load data from the CSV file into the table
        csv_file_path = '/Users/rakshit/Desktop/vgsales.csv'  # Ensure correct path
        cursor.execute(f"""
            LOAD DATA LOCAL INFILE '{csv_file_path}' 
            INTO TABLE Values_from_arduino
            FIELDS TERMINATED BY ','  
            ENCLOSED BY '"'
            LINES TERMINATED BY '\r\n'
            IGNORE 1 ROWS;
        """)

        # Commit and close the connection
        db_connection.commit()
        cursor.close()
        db_connection.close()
        
        print("Database and table setup complete. Data loaded successfully!")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    
    return True


# Step 2: Fetch data from db1 for training the ML model
def fetch_data_from_db1():
    try:
        # Connect to Database 1
        db_connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysqlgoel',
            database='db1_name'
        )

        cursor = db_connection.cursor(dictionary=True)

        # Fetch data from the table
        cursor.execute("SELECT * FROM Values_from_arduino")

        # Fetch all rows as a list of dictionaries
        data_from_db1 = cursor.fetchall()

        # Close the connection
        cursor.close()
        db_connection.close()

        return data_from_db1

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []


# Step 3: Train the ML model and make predictions
def train_ml_model(data):
    df = pd.DataFrame(data)

    # Features and target variable (adjust based on your actual data)
    features = ['accel_x', 'accel_y', 'accel_z', 'gyro_y', 'gyro_z']
    target = 'gyro_x'

    # Split data into features (X) and target (y)
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model (optional)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    return predictions


# Step 4: Insert data (including predictions) into db2
def insert_data_into_db2(data, predictions):
    try:
        # Connect to Database 2
        db_connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysqlgoel',
            database='db2_name'  # Make sure this database exists
        )

        cursor = db_connection.cursor()

        # Prepare the insert query
        insert_query = """
        INSERT INTO destination_table (accel_x, accel_y, accel_z, gyro_x, prediction)
        VALUES (%s, %s, %s, %s, %s)
        """

        # Insert original data along with predictions
        for i, row in enumerate(data):
            cursor.execute(insert_query, (
                row['accel_x'],
                row['accel_y'],
                row['accel_z'],
                row['gyro_x'],  # original target value (gyro_x)
                predictions[i]  # prediction from the ML model
            ))

        # Commit the transaction
        db_connection.commit()
        print("Data and predictions inserted successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        cursor.close()
        db_connection.close()


# Main function to orchestrate everything
def main():
    # Step 1: Set up the database and load the CSV data
    if not setup_database_and_load_data():
        print("Failed to set up the database and load data.")
        return
    
    # Step 2: Fetch the data from db1
    data_from_db1 = fetch_data_from_db1()
    
    if data_from_db1:
        # Step 3: Train the ML model and get predictions
        predictions = train_ml_model(data_from_db1)

        # Step 4: Insert the data and predictions into db2
        insert_data_into_db2(data_from_db1, predictions)
    else:
        print("No data found in db1.")

if __name__ == "__main__":
    main()
