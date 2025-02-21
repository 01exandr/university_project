clear;
clc;
% Передавальна функція розімкненої системи
% W=tf([0 10], [1 0]);
% W=tf([0 360],[1 36 10]);
W=tf([0 0.1],[0.001 1]);
% Синтез пропорційного регулятора
Cp=pidtune(W,'p')
% Синтез інтегрального регулятора
Ci=pidtune(W,'i')
% Синтез пропорційно-диференційного регулятора
Cpd=pidtune(W,'pd')
% Синтез пропорційно-інтегрального регулятора
Cpi=pidtune(W,'pi')
% Синтез пропорційно-інтегрально-диференційного регулятора
Cpid=pidtune(W,'pid')
% Передавальна функція розімкненої системи без регулятора
Wclose=feedback(W,1);
% Передавальна функція розімкненої системи з П-регулятором
Wp=feedback(series(Cp,W),1);
% Передавальна функція розімкненої системи з І-регулятором
Wi=feedback(series(Ci,W),1);
% Передавальна функція розімкненої системи з ПД-регулятором
Wpd=feedback(series(Cpd,W),1);
% Передавальна функція розімкненої системи з ПІ-регулятором
Wpi=feedback(series(Cpi,W),1);
% Передавальна функція розімкненої системи з ПІД-регулятором
Wpid=feedback(series(Cpid,W),1);
step(Wclose,'k--', Wp,Wi,Wpd,Wpi,Wpid);
legend('без регулятора', 'П-регулятор', 'І-регулятор',...
    'ПД-регулятор', 'ПІ-регулятор', 'ПІД-регулятор',...
    'Location','SouthEast');
grid;
