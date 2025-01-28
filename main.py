"""
BDA (Bangla Text Augmentation Tool) - A script to augment text data in CSV files.
"""

import pandas as pd
from augmenter import bda_augmenter

# Constants
OUTPUT_FILENAME = "augmented_data.csv"
HEADER_ART = """
╔══════════════════════════════════════╗
║      BDA - Bangla Text Augmenter     ║
╚══════════════════════════════════════╝
"""

def main():
    print(HEADER_ART)
    
    try:
        # Get and validate file path
        # file_path = input("Please provide the full path to the CSV file for augmentation: ").strip(' "')
        file_path = "F:\Current Research\BDA-test\\augmented_data.csv"
        data = load_data(file_path)
        
        # Get augmentation parameters
        column_name = get_valid_column(data)
        num_augmentations = get_valid_integer("Enter the number of augmented texts per sample: ")
        
        # Perform augmentation
        augmented_data = perform_augmentation(data, column_name, num_augmentations)
        
        # Save and show resultsc
        save_results(augmented_data)
        print_operation_summary(data, augmented_data, num_augmentations)
        
    except FileNotFoundError:
        print("Error: The specified file was not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def load_data(file_path):
    """Load and validate CSV file."""
    data = pd.read_csv(file_path)
    print("\nCSV file loaded successfully. First 5 rows:")
    print(data.head())
    return data

def get_valid_column(data):
    """Get and validate column name input."""
    while True:
        column = input("\nEnter the column name to augment: ").strip()
        if column in data.columns:
            if data[column].dtype == object:
                return column
            print(f"Error: '{column}' column must contain text data.")
        else:
            print(f"Error: Column '{column}' not found. Available columns: {list(data.columns)}")

def get_valid_integer(prompt):
    """Get and validate integer input."""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Error: Please enter a valid integer.")

def perform_augmentation(data, column_name, num_augmentations):
    """Generate augmented data while preserving original rows."""
    augmented_rows = []
    
    for _, row in data.iterrows():
        # Keep original data
        augmented_rows.append(row)
        
        # Generate augmented versions
        for _ in range(num_augmentations):
            new_row = row.copy()
            original_text = new_row[column_name]
            new_row[column_name] = bda_augmenter(original_text)
            augmented_rows.append(new_row)
    
    return pd.DataFrame(augmented_rows).reset_index(drop=True)

def save_results(dataframe):
    """Save augmented data to CSV."""
    dataframe.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\nAugmented data saved to '{OUTPUT_FILENAME}' in current directory.")

def print_operation_summary(original_df, augmented_df, num_augmentations):
    """Display operation summary statistics."""
    original_count = len(original_df)
    new_count = len(augmented_df) - original_count
    augmentation_rate = (new_count / original_count) * 100
    
    print(f"\n{' Operation Summary ':-^40}")
    print(f"Original entries: {original_count}")
    print(f"New entries created: {new_count}")
    print(f"Augmentation rate: {augmentation_rate:.1f}%")
    print(f"Total entries after augmentation: {len(augmented_df)}")
    print("\nFirst 5 augmented rows:")
    print(augmented_df.head())

if __name__ == "__main__":
    main()