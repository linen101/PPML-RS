import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_process_gas_sensor_data(data_folder="gas-sensor-array-temperature-modulation", test_percentage=0.2):
    """
    Loads and processes the Gas Sensor Array Temperature Modulation dataset.

    Parameters:
    - data_folder: Name of the folder containing the dataset (default is "gas_sensor_data").
    - test_percentage: Fraction of data to be used for testing (default is 0.2)

    Returns:
    - X_train: Training feature matrix
    - X_test: Testing feature matrix
    - Y_train: Training labels
    - Y_test: Testing labels
    """
    # Get the absolute path of the dataset directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, data_folder)

    # Initialize an empty DataFrame
    all_data = []

    # Iterate over all files in the data directory
    for filename in os.listdir(data_directory):
        if filename.endswith(".csv"):  
            file_path = os.path.join(data_directory, filename)

            try:
                # Read CSV using comma as the delimiter and ensure the first row is treated as column headers
                data = pd.read_csv(file_path, sep=",", header=0)  # Correctly handles headers
                
                # Debug column structure
                print(f"Loaded {filename}: {data.shape}")

                all_data.append(data)

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Concatenate all files into a single DataFrame
    all_data = pd.concat(all_data, ignore_index=True)

    # Ensure all values are numeric and drop invalid rows
    all_data = all_data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
    all_data = all_data.dropna()  # Remove rows with non-numeric values

    # Expected column names based on dataset documentation
    column_names = ['Time', 'CO_concentration', 'Humidity', 'Temperature', 'Flow_rate', 'Heater_voltage'] + \
                   [f'R{i}' for i in range(1, 15)]

    # Ensure that all_data has the expected number of columns
    #if all_data.shape[1] != len(column_names):
    #    print(f"Error: Column mismatch! Data has {all_data.shape[1]} columns, expected {len(column_names)}.")
    #    print("First few rows of data:")
    #    print(all_data.head())
    #    return None

    # Assign correct column names
    all_data.columns = column_names

    # Select features (sensor resistances) and target (CO concentration)
    #X = all_data[[f'R{i}' for i in range(1, 15)]].values
    # Define the selected features explicitly
    selected_features = ['T5', 'R2', 'R3', 'R4']

    # Select only the specified features
    X = all_data[selected_features].values

    Y = all_data['CO_concentration'].values

    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_percentage, random_state=42)

    return X_train, X_test, Y_train, Y_test


def load_and_process_gas_sensor_data_dynamic_mixture(file_path="/home/linen/Desktop/PhD/project robust stats copy/robust-statistics-internal/main/realdata/gas+sensor+array+under+dynamic+gas+mixtures/ethylene_CO.txt"):
    try:
        # Attempt to read with space/tab as delimiter
        df = pd.read_csv(file_path, sep=r'\s+', engine='python', header=None)
        
        # Check column count
        if df.shape[1] != 19:
            print(f"Unexpected column count: {df.shape[1]} (Expected: 19). Retrying with auto-detected delimiter.")
            
            # Detect delimiter automatically
            with open(file_path, 'r') as f:
                dialect = csv.Sniffer().sniff(f.read(1000))
                detected_delimiter = dialect.delimiter
                print(f"Detected delimiter: '{detected_delimiter}'")

            df = pd.read_csv(file_path, delimiter=detected_delimiter, header=0)

        # Verify if the issue persists
        if df.shape[1] != 19:
            print(f"Still incorrect number of columns: {df.shape[1]}. Exiting.")
            return None

        # Rename columns correctly
        columns = ['Time', 'CO_conc', 'Ethylene_conc'] + [f'Sensor_{i+1}' for i in range(16)]
        df.columns = columns

        # Handle sensor data transformation (Avoid division by zero)
        sensor_cols = [f'Sensor_{i+1}' for i in range(16)]
        
        # Convert sensor readings to KOhms (avoid negative values)
        df[sensor_cols] = df[sensor_cols].applymap(lambda x: 40000 / abs(x) if x != 0 else 0)

        # Select all features
        X = df[sensor_cols]
        # Select features for regression
        #selected_features = ['Sensor_3', 'Sensor_4', 'Sensor_5']  # You can modify based on importance
        #X = df[selected_features]  
        y = df['Ethylene_conc']  

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    


def read_data(file_path, input_columns, target_column):
    X_data = [[] for _ in range(len(input_columns))]
    Y_data = []

    with open(file_path, newline='') as csvfile:
        energy_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        first = True
        for row in energy_reader:
            if first:
                first = False
            else:
                for i, col in enumerate(input_columns):
                    X_data[i].append(float(row[col]))
                Y_data.append(float(row[target_column]))

    X = np.column_stack([np.array(col) for col in X_data])
    Y = np.array(Y_data).reshape(-1, 1)
    return X, Y

def load_and_process_energy_data(test_percentage=0.2):
    file_path = '/home/ubuntu/PPML-RS/main/realdata/energydata_complete.csv'
    #file_path = '/home/linen/Desktop/PhD/project robust stats copy/robust-statistics-internal/main/realdata/energydata_complete.csv'
    input_columns = [3, 5, 7, 9, 11, 13, 15]
    #input_columns = [11,13,15]
    target_column = 19

    X, Y = read_data(file_path, input_columns, target_column)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_percentage)
    
    return X_train, X_test, Y_train, Y_test 
    

# Example usage
if __name__ == "__main__":

    X_train, X_test, Y_train, Y_test = load_and_process_energy_data()
    print("Training data shape:", X_train.shape, Y_train.shape)
    print("Testing data shape:", X_test.shape, Y_test.shape)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Predict on test set
    Y_pred = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(Y_test, Y_pred)
    print(f"Mean Absolute Error: {mae:.3f}")

    # Plot predictions vs. actual
    plt.scatter(Y_test, Y_pred, alpha=0.7, colorizer='violet')
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title('Actual vs. Predicted Temperature on a clean dataset')
    plt.show()
