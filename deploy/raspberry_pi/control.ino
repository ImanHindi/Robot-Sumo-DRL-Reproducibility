/********* PIN MAP: ARDUINO UNO *********/
// BTS7960 — Left motor
const int L_EN_L   = 12;   // optional: tie to 5V if you want
const int L_EN_R   = 13;   // optional: tie to 5V if you want
const int L_LPWM   = 5;    // LPWM (PWM)
const int L_RPWM   = 6;    // RPWM (PWM)

// BTS7960 — Right motor
const int R_EN_L   = A0;   // optional: tie to 5V if you want
const int R_EN_R   = A1;   // optional: tie to 5V if you want
const int R_LPWM   = 9;    // LPWM (PWM)
const int R_RPWM   = 10;   // RPWM (PWM)

// HC-SR04 (front/left/right)
const int TRIG_F = 2,  ECHO_F = 3;   // Front
const int TRIG_L = A3, ECHO_L = A4;  // Left
const int TRIG_R = 11, ECHO_R = A2;  // Right

// Digital sensors
const int BACK_PIN   = 8;   // back IR obstacle (digital)
const int LINE_L_PIN = 4;   // line/edge left (digital)
const int LINE_R_PIN = 7;   // line/edge right (digital)

/********* LOGIC INVERSION (flip if needed) *********/
const bool BACK_ACTIVE_LOW  = true;   // many obstacle modules LOW on detect
const bool LINE_ACTIVE_LOW  = true;   // line sensor LOW on black/edge

/********* OTHER CONFIG *********/
const long BAUD = 115200;

int dutyFromPercent(int p) {
  p = constrain(p, -100, 100);
  return map(abs(p), 0, 100, 0, 255);
}

void bts7960Init() {
  pinMode(L_EN_L, OUTPUT); pinMode(L_EN_R, OUTPUT);
  pinMode(R_EN_L, OUTPUT); pinMode(R_EN_R, OUTPUT);
  pinMode(L_LPWM, OUTPUT); pinMode(L_RPWM, OUTPUT);
  pinMode(R_LPWM, OUTPUT); pinMode(R_RPWM, OUTPUT);

  // enable drivers (or hard-wire EN pins to 5V)
  digitalWrite(L_EN_L, HIGH);
  digitalWrite(L_EN_R, HIGH);
  digitalWrite(R_EN_L, HIGH);
  digitalWrite(R_EN_R, HIGH);
}

void motorsStop() {
  analogWrite(L_LPWM, 0); analogWrite(L_RPWM, 0);
  analogWrite(R_LPWM, 0); analogWrite(R_RPWM, 0);
}

void setMotorLR_percent(int leftPct, int rightPct) {
  leftPct  = constrain(leftPct,  -100, 100);
  rightPct = constrain(rightPct, -100, 100);

  // Left
  if (leftPct >= 0) { analogWrite(L_LPWM, dutyFromPercent(leftPct)); analogWrite(L_RPWM, 0); }
  else              { analogWrite(L_LPWM, 0);                         analogWrite(L_RPWM, dutyFromPercent(leftPct)); }

  // Right
  if (rightPct >= 0) { analogWrite(R_LPWM, dutyFromPercent(rightPct)); analogWrite(R_RPWM, 0); }
  else               { analogWrite(R_LPWM, 0);                          analogWrite(R_RPWM, dutyFromPercent(rightPct)); }
}

// HC-SR04 distance in cm (utility)
float readHCSR04cm(int trig, int echo) {
  digitalWrite(trig, LOW); delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long dur = pulseIn(echo, HIGH, 30000UL); // timeout ~30ms
  if (dur == 0) return 400.0; // no echo
  float cm = (dur / 2.0f) / 29.1f;
  if (cm < 0) cm = 0;
  if (cm > 400) cm = 400;
  return cm;
}

int readDigitalFlag(int pin, bool activeLow) {
  int v = digitalRead(pin);
  return activeLow ? ((v == LOW) ? 1 : 0) : ((v == HIGH) ? 1 : 0);
}

void sendSensors() {
  float front_cm = readHCSR04cm(TRIG_F, ECHO_F);
  float left_cm  = readHCSR04cm(TRIG_L, ECHO_L);
  float right_cm = readHCSR04cm(TRIG_R, ECHO_R);
  int backFlag   = readDigitalFlag(BACK_PIN,   BACK_ACTIVE_LOW);
  int lineLFlag  = readDigitalFlag(LINE_L_PIN, LINE_ACTIVE_LOW);
  int lineRFlag  = readDigitalFlag(LINE_R_PIN, LINE_ACTIVE_LOW);

  // Packet: S,front_cm,back_flag,left_cm,right_cm,lineL_flag,lineR_flag
  Serial.print("S,");
  Serial.print(front_cm, 1);  Serial.print(",");
  Serial.print(backFlag);     Serial.print(",");
  Serial.print(left_cm, 1);   Serial.print(",");
  Serial.print(right_cm, 1);  Serial.print(",");
  Serial.print(lineLFlag);    Serial.print(",");
  Serial.println(lineRFlag);
}

String inbuf;

void setup() {
  Serial.begin(BAUD);

  pinMode(TRIG_F, OUTPUT); pinMode(ECHO_F, INPUT);
  pinMode(TRIG_L, OUTPUT); pinMode(ECHO_L, INPUT);
  pinMode(TRIG_R, OUTPUT); pinMode(ECHO_R, INPUT);

  pinMode(BACK_PIN,   INPUT_PULLUP);
  pinMode(LINE_L_PIN, INPUT_PULLUP);
  pinMode(LINE_R_PIN, INPUT_PULLUP);

  bts7960Init();
  motorsStop();
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      inbuf.trim();
      if (inbuf.length() > 0) {
        if (inbuf == "R") {
          sendSensors();
        } else if (inbuf.startsWith("V,")) {
          int c1 = inbuf.indexOf(',');
          int c2 = inbuf.indexOf(',', c1 + 1);
          if (c2 > 0) {
            int L = inbuf.substring(c1 + 1, c2).toInt();
            int R = inbuf.substring(c2 + 1).toInt();
            setMotorLR_percent(L, R);
          }
        } else if (inbuf == "S") {
          motorsStop();
        }
      }
      inbuf = "";
    } else if (c != '\r') {
      inbuf += c;
    }
  }
}
