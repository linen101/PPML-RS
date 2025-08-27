

# Expected column names based on dataset documentation
    #column_names = ['Type', 'Name', 'Age', 'Breed1', 'Breed2', 
    #                'Gender' , 'Color1', 'Color2', 'Color3', 
    #                'MaturitySize', 'FurLength' , 'Vaccinated',
    #                'Dewormed', 'Sterilized', 'Health', 'Quantity',
    #                'Fee', 'State', 'RescuerID', 'VideoAmt' , 
    #                'Description', 'PetID', 'PhotoAmt' , 'AdoptionSpeed'] 
import sys    
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
#module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
#if module_path not in sys.path:
#    sys.path.append(module_path)
    
#from torrent.torrent import  torrent    

def load_and_process_pet_finder_data(data_folder="pet_finder"):
    """
    Loads and processes Pet Finder dataset, keeping only cats (Type = 2).
    Splits data into training and testing sets.

    Returns:
    - X_train, X_test: Feature matrices for training and testing
    - Y_train, Y_test: Labels for training and testing
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, data_folder)

    all_data = []

    for filename in os.listdir(data_directory):
        if filename.endswith(".csv"):  
            file_path = os.path.join(data_directory, filename)
            try:
                data = pd.read_csv(file_path, sep=",", header=0)
                print(f"Loaded {filename}: {data.shape}")
                all_data.append(data)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not all_data:
        print("No CSV files loaded. Check the data folder path and file format.")
        return None

    # Concatenate all DataFrames into a single DataFrame
    all_data = pd.concat(all_data, ignore_index=True)

    # Debug: Check initial dataset
    print("Initial Data Info:")
    print(all_data.info())
    print(all_data.head()) 

    # Identify missing values before dropping
    print("Missing values per column before dropna:")
    print(all_data.isnull().sum())

    # **Filter only cats (Type == 2)**
    #all_data = all_data[all_data['Type'] == 2]
    #print(f"Filtered dataset for cats only: {all_data.shape}")

    # Define selected features and target
    selected_features = ['FurLength', 'Age', 'Fee', 'State',  'VideoAmt' , 
                         'PhotoAmt', 'Quantity', 'Vaccinated',
                         'Dewormed', 'Sterilized', 'MaturitySize']  # Modify this list based on available features
    label_column = 'AdoptionSpeed'

    # Drop NaN values only from necessary columns
    required_columns = selected_features + [label_column]
    rows_before = all_data.shape[0]
    all_data = all_data.dropna(subset=required_columns)
    rows_after = all_data.shape[0]

    print(f"Rows before filtering: {rows_before}, Rows after filtering: {rows_after}")

    if all_data.empty:
        print("No valid data after filtering. Check CSV contents.")
        return None

    # Convert to correct data types
    all_data[selected_features] = all_data[selected_features].apply(pd.to_numeric, errors='coerce')
    all_data[label_column] = all_data[label_column].astype(int)

    # Extract features and target
    X = all_data[selected_features].values
    Y = all_data[label_column].values

    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test


# Example usage
if __name__ == "__main__":
    result = load_and_process_pet_finder_data()
    
    if result:
        X_train, X_test, Y_train, Y_test = result
        print("Training data shape:", X_train.shape, Y_train.shape)
        print("Testing data shape:", X_test.shape, Y_test.shape)

        # Train model on only cats
        #model = torrent(X_train, Y_train, 0.05 , 0.1)
        #LinearRegression()
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # Predict on test set
        Y_pred = model.predict(X_test)

        # Evaluate model
        mae = mean_absolute_error(Y_test, Y_pred)
        print(f"Mean Absolute Error: {mae:.3f}")
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = (mse ** 0.5) / 4  # Root Mean Squared Error divided by the range, wishing for rmse 0.1 -0.2
        print(f"Normalized Mean Square Error: {rmse:.3f}")
         # Plot predictions vs. actual
        plt.scatter(Y_test, Y_pred, alpha=0.7)
        plt.xlabel("Actual Adoption Time")
        plt.ylabel("Predicted Adoption Time")
        y_max =  4
        plt.yticks(np.arange(0, y_max + 1, 1)) 
        plt.title("Actual vs. Predicted Adoption Time")
        plt.show()