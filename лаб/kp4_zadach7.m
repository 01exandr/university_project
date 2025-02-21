clear;
clc;
% pkg load control
% Передавальна функція розімкненої системи
W=tf([0 0 30],[1 18.6 54])
% Передавальна функція розімкненої системи
Wclose=feedback(W, 1)
figure(1);
step(Wclose);
hold on;
% Перетворення безперервної функції у дискретну
WDclose=c2d(Wclose,0.1)
step(WDclose);
hold off;
grid;
figure(2);
hold on;
% Побудова годографа Михайлова
T=0.1;
for w=0:0.1:10*pi
    A = 99850*exp(0.2*j*w)-83410*exp(0.1*j*w)+20110;
    P=real(A);
    Q=imag(A);
    plot(P,Q,'k.');
end
hold off;
grid;
title('Годограф Михайлова для дискретної системи');
WDopen=c2d(W,0.1);
figure(3);
% Побудова годографа Найквиста
nyquist(WDopen);
grid;

