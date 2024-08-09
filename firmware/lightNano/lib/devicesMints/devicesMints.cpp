
#include "devicesMints.h"

// AS7265X ---------------------------------------
bool initializeAS7265X(){
  Serial.println("Attemmpting to connect to : AS7265X" );
  bool status = false;
  for (int i = 0; i <= 255; i++) {
    Serial.print("Attempt: " );
    Serial.println(i);
    status  = as7265x.begin();
    delay(1000);
    if(status){
      Serial.println("Connected to AS7265x");
      break;
    }
  }

  if (status) {
    as7265x.disableIndicator();
    delay(1000);
    Serial.println("AS7265x Initiated");
    return true;
  }else{
    Serial.println("AS7265x not found");
    delay(1);
    return false;
  }
}

void readAS7265X(){
  // Serial.println("AS7265X reader");
  String sensorName = "AS7265X" ;

  as7265x.takeMeasurements();
  String readings[18]  = {
                      String(as7265x.getCalibratedA()),
                      String(as7265x.getCalibratedB()),
                      String(as7265x.getCalibratedC()),
                      String(as7265x.getCalibratedD()),
                      String(as7265x.getCalibratedE()),
                      String(as7265x.getCalibratedF()),
                      String(as7265x.getCalibratedG()),
                      String(as7265x.getCalibratedH()),
                      String(as7265x.getCalibratedR()),
                      String(as7265x.getCalibratedI()),
                      String(as7265x.getCalibratedS()),
                      String(as7265x.getCalibratedJ()),
                      String(as7265x.getCalibratedT()),
                      String(as7265x.getCalibratedU()),
                      String(as7265x.getCalibratedV()),
                      String(as7265x.getCalibratedW()),
                      String(as7265x.getCalibratedK()),
                      String(as7265x.getCalibratedL()),
                      };
  
  sensorPrintMints("AS7265X",readings,18);

  }

bool initializeLTR390V2(){
  bool status = ltr390.begin(); 
  if (status) {
    Serial.println("Found LTR sensor!");

    ltr390.setMode(LTR390_MODE_UVS);
    if (ltr390.getMode() == LTR390_MODE_ALS) {
      Serial.println("In ALS mode");
    } else {
      Serial.println("In UVS mode");
    }

    ltr390.setGain(LTR390_GAIN_3);
    Serial.print("Gain : ");
    switch (ltr390.getGain()) {
      case LTR390_GAIN_1: Serial.println(1); break;
      case LTR390_GAIN_3: Serial.println(3); break;
      case LTR390_GAIN_6: Serial.println(6); break;
      case LTR390_GAIN_9: Serial.println(9); break;
      case LTR390_GAIN_18: Serial.println(18); break;
    }

    ltr390.setResolution(LTR390_RESOLUTION_16BIT);
    Serial.print("Resolution : ");
    switch (ltr390.getResolution()) {
      case LTR390_RESOLUTION_13BIT: Serial.println(13); break;
      case LTR390_RESOLUTION_16BIT: Serial.println(16); break;
      case LTR390_RESOLUTION_17BIT: Serial.println(17); break;
      case LTR390_RESOLUTION_18BIT: Serial.println(18); break;
      case LTR390_RESOLUTION_19BIT: Serial.println(19); break;
      case LTR390_RESOLUTION_20BIT: Serial.println(20); break;
    }

    ltr390.setThresholds(100, 1000);
    ltr390.configInterrupt(true, LTR390_MODE_UVS);
    
    Serial.println("Found LTR sensor!");
    Serial.println("LTR390V2 Initiated");
    return true;
  }else{
    Serial.println("LTR390V2 not found");
    delay(1);
    return false;
  }
}



void readLTR390V2(){
  // Serial.println("AS7265X reader");
  String sensorName = "LTR390V2" ;
  uint32_t als,uvs;
  bool alsRead, uvsRead = 0;
  
  
  ltr390.setMode(LTR390_MODE_ALS);
  delay(100);
  if (ltr390.newDataAvailable())
  { 
    alsRead = 1;
    als = ltr390.readALS();
  }

  delay(100);

  ltr390.setMode(LTR390_MODE_UVS);
  delay(100);
  if (ltr390.newDataAvailable())
  { 
    uvs = ltr390.readUVS();
    uvsRead = 1;
  }

  String readings[4]  = {
                      String(alsRead),
                      String(als),
                      String(uvsRead),
                      String(uvs),
  };

  sensorPrintMints("LTR390V2",readings,4);

}


// INS219s ---------------------------------------
bool initializeGUVAS12SDV2(){

  bool INA219Status = ina219.begin();

  if (INA219Status) {
      Serial.println("INA219 Initiated");
  }else{
    Serial.println("INA219 not found");
    delay(1);
  }

  return INA219Status;
}


void readGUVAS12SDV2(){
  String sensorName = "GUVAS12SDV2" ;
  String readings[2]  = {
                      String(ina219.getShuntVoltage_mV()),
                      String(ina219.getBusVoltage_V()),
    };
    sensorPrintMints("GUVAS12SDV2",readings,2);
  }
