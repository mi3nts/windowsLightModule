import pandas as pd
from collections import OrderedDict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import time
import datetime
from mintsXU4 import mintsDefinitions as mD
from mintsXU4 import mintsSensorReader as mSR

# Path to the CSV file
monitor_directory = '../mintsData/MINTS_Minolta_10004098_2024_08_27_22.csv'

nodeID   = mD.macAddress
sensorID = "MIN001"


import pandas as pd
import datetime
from collections import OrderedDict

def csv_to_ordered_dict(csv_file):
    # Read the CSV file and get the last row
    df = pd.read_csv(csv_file).tail(1)
    # Clean column names by removing spaces and slashes
    df.columns = df.columns.str.replace(' ', '').str.replace('/', '').str.replace(']', '').str.replace('[', '')

    # Create a combined datetime string and parse it
    date_time_str = f"{df['Date'].iloc[0]} {df['Time'].iloc[0]}"
    date_time = datetime.datetime.strptime(date_time_str, '%Y/%m/%d %H:%M:%S')

    # Format datetime with microseconds
    date_time_str = date_time.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Drop original 'Date' and 'Time' columns
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Add the new 'dateTime' column at the front
    df['dateTime'] = date_time_str
    df = df[['dateTime'] + [col for col in df.columns if col != 'dateTime']]

    # Create a mapping for renaming 'Spectrum[i]' columns to 'channelXXXnm'
    spectrum_columns = {f'Spectrum{i}': f'channel{360 + i}nm' for i in range(421)}

    # Rename the DataFrame columns
    df.rename(columns=spectrum_columns, inplace=True)

    # Convert the DataFrame to a list of dictionaries and then to an OrderedDict
    records = df.to_dict(orient='records')
    ordered_dict = OrderedDict((k, v) for record in records for k, v in record.items())

    return date_time, ordered_dict

# class ChangeHandler(FileSystemEventHandler):
#     def on_modified(self, event):
#         # Check if the modified file is the target CSV file
#         if event.src_path == os.path.abspath(csv_file):
#             print("File changed, processing...")
#             dateTime, sensorDictionary = csv_to_ordered_dict(csv_file)
#             mSR.sensorFinisher(dateTime,sensorID,sensorDictionary)

# def monitor_file(file_path):
#     event_handler = ChangeHandler()
#     observer = Observer()
#     observer.schedule(event_handler, path=os.path.dirname(file_path), recursive=False)
    
#     print("Monitoring file for changes...")
#     observer.start()

#     try:
#         while True:
#             time.sleep(1)  # Keep the script running
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()

# # Start monitoring the CSV file for changes
# monitor_file(csv_file)


class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # Ensure the modified file is a CSV file
        if event.src_path.endswith('.csv'):
            print(f"File changed: {event.src_path}, processing...")
            try:
                # Process the specific CSV file
                dateTime, sensorDictionary = csv_to_ordered_dict(event.src_path)
                mSR.sensorFinisher(dateTime, sensorID, sensorDictionary)
            except Exception as e:
                print(f"Error processing file {event.src_path}: {e}")

def monitor_directory_for_csv_changes(directory_path):
    event_handler = ChangeHandler()
    observer = Observer()
    # Set recursive=True to monitor all subdirectories as well
    observer.schedule(event_handler, path=directory_path, recursive=True)
    
    print(f"Monitoring directory '{directory_path}' and its subdirectories for changes in CSV files...")
    observer.start()

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        print("Stopping monitoring...")
        observer.stop()
    observer.join()

# Start monitoring the parent directory for changes in CSV files
monitor_directory_for_csv_changes(monitor_directory)
