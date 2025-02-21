%W=tf([0 0 76 -760],[10 1 0 0]);
pkg load control
%
% Критерій Михайлова
%W=tf([69.984*10^(-6) 1086.48*10^(-4) 1],[69.984*10^(-6) 1105.92*10^(-4) 1]);
%W0=tf([0 0 36],[1 3.6 72]);
%Wclose=feedback(W,W0);
Wclose=tf([0 10 -10],[1 9 -110])
figure(1);
for w=0:0.1:200
   A=1*(1i*w)^2+9*(1i*w)^1-110;
   P=real(A);
   Q=imag(A);
   plot(P,Q,'r.');
   hold on;
end
hold off;
grid;

figure(2);
step(Wclose);
grid;
%

%{
% Критерій Найквіста
%W=tf([0 0 0 0.36],[0.1 0.2 1 0]);
W=tf([0 0 175000 -1750],[0.01 1 0 0]);
W0=tf([0 0 0 1],[0 0 0 1]);
Wopen=series(W,W0);
figure(3);
nyquist(Wopen);
grid;

% Bode Diagram
W0=tf([0 0 0 35],[0 0 0 1]);
Wopen=series(W,W0);
figure(4);
margin(Wopen);
grid;
%}
