# windowsLightModule
Contains firmware to run windows sw to collect light sensor data 


# Run Intuctions 
- Log in through DWService: Mints Light Sensing Unit   
- Make sure the time is in GMT 
- On the desktop double click on Minolta data parser and press enter 
- Open 3 powershells and run the following
  - cd .\Desktop\gitHubRepos\WindowsLightModule\firmware\xu4Mqtt ; python nanoReader.py 0
  - cd .\Desktop\gitHubRepos\WindowsLightModule\firmware\xu4Mqtt ; python GPSReader.py
  - cd .\Desktop\gitHubRepos\WindowsLightModule\firmware\xu4Mqtt ; python skyCamReaderSaveOnly.py
- Data is available on the desktop under the folders Minolts and mintsData. Copy the data into a zip file and download through DWService until rclone is available. 
