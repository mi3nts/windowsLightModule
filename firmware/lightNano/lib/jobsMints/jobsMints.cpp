#include "jobsMints.h"

void initializeSerialMints(){
    Serial.begin(9600);
    Serial.println("Serial Port Open");
}
//
// void initializeSerialUSBMints(){
//     SerialUSB.begin(9600);
//     Serial.println("SerialUSB Port Open");
// }

void sensorPrintMints(String sensor,String readings[],uint8_t numOfvals){
  Serial.print("#mintsO!");Serial.print(sensor);Serial.print(">");
  for (int i = 0; i < numOfvals; ++i)
      {
        Serial.print(readings[i]);Serial.print(":");
      }
      Serial.print("~");
}

void delayMints(unsigned int timeSpent,unsigned int loopInterval){
  unsigned int loopIntervalReal = loopInterval+ 30 ;
  unsigned int waitTime;
  if(loopIntervalReal>timeSpent){
    waitTime = loopIntervalReal - timeSpent;
    delay(waitTime);
  }

}

// Native Command
// void serialEvent() {
//   while (Serial.available()) {
//     // get the new byte:
//     char inChar = (char)Serial.read();
//     // add it to the inputString:
//
//     inputString += inChar;
//     // if the incoming character is a newline, set a flag so the main loop can
//     // do something about it:
//     if (inChar == '-') {
//        stringComplete = true;
//     }
//   }
// }
//

//
//
// void commandReadMints(){
//     // Serial.println("inside");
//     if (stringComplete) {
//       Serial.println(inputString);
//       sendCommand2DevicesMints(inputString);
//
//       // clear the string:
//       inputString = "";
//       stringComplete = false;
//     }
// }
//
// void sendCommand2DevicesMints(String command){
//
//       if (command.startsWith("mints:")) {
//         printInput("Recieving Mints Command");
//
//       if (command.startsWith("HTU21D",6)) {
//         readHTU21DMints();
//       }
//
//       if (command.startsWith("BMP280",6)) {
//         readBMP280Mints();
//         // SerialUSB.println(year());
//       }
//
//       if (command.startsWith("time",6)) {
//         setTimeMints(command.substring(11));
//
//         }
//
//     }
// }
//


// String int2StringMints(int inputNumber){
//
// return String::format("%04d:%02d:%02d:%02d:%02d:%02d",inputNumber);
//
//   }
