#include <Stepper.h>
#define steps 10 // Кількість кроків на один оберт
// сонар
#define ECHO 2
#define TRIG 3
//time
#include <TimerOne.h>
//бібліотека сонара
#include <NewPing.h>
NewPing sonar(TRIG, ECHO, 400);

float point = 5;
float rezult = 0;
float dist_3[3] = {0.0, 0.0, 0.0};	// массив для зберігання трьох останніх вимірів
float middle, dist, dist_filtered, len;
float k;
byte i, delta;
unsigned long sensTimer;

// піни Arduino, до яких підключені обмотки двигуна
Stepper myStepper(steps, 7, 6, 5, 4);

void setup() {
  Serial.begin(9600);
  Serial.println("task_poit, input, input_filter");
  // встановлюємо швидкість обертання об./хв.
  myStepper.setSpeed(60); 
}

void loop() {
  // вимір і вивід кожні 500 мс
  if (millis() - sensTimer > 500) { 
    Serial.print(point); //task_point
    Serial.print(",");
    creation_arr(dist_3);   
    Serial.print(dist_3[1]); // input
    Serial.print(",");

    dist_filtered = middle_of_dist(dist_3[0], dist_3[1], dist_3[2]);
    Serial.println(dist_filtered); // input_filter

    rezult = computePID(dist_filtered, point, 1.5, 1.5, 0, 0.05, 2, 6);
    // умова з визначенням руху сонару
    if (dist_filtered - point > 0.1 ){
      myStepper.step(rezult*steps);
    }
    else if(dist_filtered - point < -0.1 ) {
      myStepper.step(rezult*(-steps));
    }
    else {myStepper.step(0);}
      ensTimer = millis();	// скинути таймер
  }
}

// (вхід, установка, п, і, д, період в секундах, мін.вихід, макс. вихід) 
int computePID(float input, float setpoint, float kp, float ki, float kd, float dt, int minOut, int maxOut) {
  float err = setpoint - input; 
  static float integral = 0, prevErr = 0; 
  integral = constrain(integral + (float)err * dt * ki, minOut, maxOut); 
  float D = (err - prevErr) / dt; 
  prevErr = err;
  return constrain(err * kp + integral + D * kd, minOut, maxOut); 
}
void creation_arr(float arr[]){
  if (i > 1) i = 0;
  else i++;
  //отримати відстань до масиву
  arr[i] = (float)sonar.ping() / 57.5; 
}

// медіанний фільтр із трьох значень
float middle_of_3(float a, float b, float c) {
  if ((a <= b) && (a <= c)) {
    middle = (b <= c) ? b : c;
  }
  else {
    if ((b <= a) && (b <= c)) {
      middle = (a <= c) ? a : c;
    }
    else {
      middle = (a <= b) ? a : b;
    }
  }
  return middle;
}

float middle_of_dist(float a, float b, float c){
// фільтрувати медіанним фільтром із трьох останніх вимірів
dist = middle_of_3(dist_3[0], dist_3[1], dist_3[2]);
// розрахунок вимірів із попереднім
delta = abs(dist_filtered - dist);
// якщо більше то різкий коефіцієнт
if (delta > 1) k = 0.7;
// якщо менше то плавний коефіцієнт
else k = 0.1;
// фільтр "рухоме середнє"
dist_filtered = dist * k + dist_filtered * (1 - k);
return dist_filtered;

