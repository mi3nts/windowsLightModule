#ifndef DEVICES_MINTS_H
#define DEVICES_MINTS_H
//
// #include <Arduino.h>
#include "jobsMints.h"
#include "SparkFun_AS7265X.h"
#include "Adafruit_INA219.h"
#include "Adafruit_LTR390.h"

extern AS7265X as7265x;
bool initializeAS7265X();
void readAS7265X();

extern Adafruit_LTR390 ltr390;
bool initializeLTR390V2();
void readLTR390V2();

extern Adafruit_INA219 ina219;
bool initializeGUVAS12SDV2();
void readGUVAS12SDV2();

#endif
