%{
This explains the plotting functions
%}

x = 1:10;
y = x.^2;  % elementwise product
disp(x);
disp(y);

% plot x and y as 2 signals on same plot
% color code, marker and line type are specified as 'rx-'
% r = red, x= marker, - = line dashed
% colors: r, b, c, m, k
% markers: *, -, ., d
% lines: -, ., -- etc.
plot(x, 'rx-')
hold on  % wihtout this the sencond plot overwrites the 1st one
plot(y, 'bx-')
hold off

% alternatively
plot(x, y, 'r*');
title("Exponential")
xlabel("x->");
ylabel("x^2->");

% making a simple sine wave
samples = 1:100;
F = 50;
Fs = 22*F;
ts = 1/Fs;
T = 1 / F;

t = 0 : ts : 2*T;
disp(t)

sines = sin(2*pi*F*t);
plot(t, sines);

coses = cos(2*pi*F*t);

% plotting two different figures in 2 windows
fig1 = figure(1);
plot(t, sines, "c*--");
title("Sine wave");
xlabel("t(ms)");
ylabel("sin");
axis padded

fig2 = figure(2);
plot(coses, "r*-");
title("Cos wave");
xlabel("t(ms)");
ylabel("cos");
axis padded