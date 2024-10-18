import os
import csv

def add_column_to_csv(input_folder, output_folder, input_text):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)

            # Read the existing CSV file and store the data in a list
            with open(input_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                data = list(reader)

            # Add the new column header
            data[0].append('label')

            # Add the input_text value to each row in the new column
            for row in data[1:]:
                row.append(input_text)

            # Write the updated data back to the output CSV file
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)

# Usage:
input_folder_path = './all_data/skating'
output_folder_path = './labelData'
input_text = 'skating'

add_column_to_csv(input_folder_path, output_folder_path, input_text)
