clear;
clc;
w1=tf([0 36], [1 0]);
w2=tf([0 0.01], [0.1 0.2 1]);
w=series(w1,w2)   % ���̦����� �'������� w1 �� w2
figure(1);
nyquist(w); % ���� - Ħ������ ����צ���
figure(2);
bode(w); % ����, ���� - Ħ������ �����
grid;