#include "Arduino.h"



// #include "Seeed_BME280.h"
// #include "MutichannelGasSensor.h"
// #include "OPCN2NanoMints.h"
#include "jobsMints.h"
#include "devicesMints.h"
#include <Wire.h>

AS7265X as7265x;
bool AS7265XOnline;

Adafruit_LTR390 ltr390 = Adafruit_LTR390();
bool LTR390V2Online;

Adafruit_INA219 ina219(0x40);
bool GUVAS12SDV2Online;


uint16_t sensingPeriod = 1000;
uint16_t initPeriod = 1500;

unsigned long startTime;

void setup() {
  delay(2000);
  
  initializeSerialMints();
  Serial.println();
  Serial.println("==========================================");
  Serial.println("=========== MINTS LIGHT NODE =============");
  Serial.println("==========================================");
  
  AS7265XOnline  = initializeAS7265X();
  delay(initPeriod);
  LTR390V2Online   = initializeLTR390V2();
  delay(initPeriod);
  GUVAS12SDV2Online   = initializeGUVAS12SDV2();
  delay(initPeriod);
}


// the loop routine runs over and over again forever:
void loop() {

    startTime  = millis();

    // delay(sensingPeriod);
    if(AS7265XOnline)
    {
      readAS7265X();
    }

    delay(sensingPeriod);
    if(LTR390V2Online)
    {
      readLTR390V2();
    }
    // //
    delay(sensingPeriod);
    if(GUVAS12SDV2Online)
    {
      readGUVAS12SDV2();
    }
    delayMints(millis() - startTime,10000);

}


