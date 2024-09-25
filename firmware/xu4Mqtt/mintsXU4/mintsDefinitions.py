
from getmac import get_mac_address
import serial.tools.list_ports

def findPort(find):
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        currentPort = str(p)
        if(currentPort.endswith(find)):
            return(currentPort.split(" ")[0])

        
def findNanoPorts():
    ports = list(serial.tools.list_ports.comports())
    nanoPorts = []
    for p in ports:
        nanoPort = str(p[2])
        if(nanoPort.find("PID=0403")>=0):
            nanoPorts.append(str(p[0]).split(" ")[0])
    return nanoPorts        


        
def findGPSPorts():
    ports = list(serial.tools.list_ports.comports())
    nanoPorts = []
    for p in ports:
        nanoPort = str(p[2])
        if(nanoPort.find("PID=1546:01A7")>=0):
            nanoPorts.append(str(p[0]).split(" ")[0])
    return nanoPorts    
  
def findAirmarPort():
    ports = list(serial.tools.list_ports.comports())
    ozonePort = []
    for p in ports:
        currentPort = str(p[2])
        if(currentPort.find("PID=067B")>=0):
            ozonePort.append(str(p[0]).split(" ")[0])
    return ozonePort
  

def findMacAddress():
    # List of potential interfaces to check
    interfaces = ["Ethernet", "Wi-Fi", "docker0", "eth0", "enp1s0", "en0", "en1", "en2", "wlan0"]
    
    for interface in interfaces:
        macAddress = get_mac_address(interface=interface)
        if macAddress is not None:
            return macAddress.replace(":", "")
    
    return "xxxxxxxx"




dataFolderReference       = "C:/Users/yichao/Desktop/mintsData/reference"
dataFolderMQTTReference   = "C:/Users/yichao/Desktop/mintsData/referenceMQTT"
dataFolder                = "C:/Users/yichao/Desktop/mintsData/raw"
dataFolderMQTT            = "C:/Users/yichao/Desktop/mintsData/rawMQTT"

nanoPorts             = findNanoPorts()
latestOn              =True
macAddress            = findMacAddress()
airmarPort            = findAirmarPort()
# For MQTT 
mqttOn                   = True
mqttCredentialsFile      = 'mintsXU4/credentials.yml'
mqttBroker               = "mqtt.circ.utdallas.edu"
mqttPort                 =  8883  # Secure port


gpsPorts                 = findGPSPorts()


if __name__ == "__main__":

    print("Mac Address: {0}".format(macAddress))

    print("Nano Ports:")
    for dev in nanoPorts:
        print("\t{0}".format(dev))

    print("GPS Ports:")
    for dev in gpsPorts:
        print("\t{0}".format(dev))