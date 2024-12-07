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

% plot exp(x) and log(x) for x = [0, 3.5]
x_vals = 0:0.5:3.5;
exp_x = exp(x);
log_x = log(x);
plot(exp_x, "ko-");
hold on
plot(log_x, "g.--");
hold off

% file save and load for data and text format
% saving and loading generally happends for matrix type data
% should have same number of rows and cols
t = 0 : ts: T;
sinex = sin(2*pi*F*t);
t_sinet = [t; sinex];  % concat along columns
disp(size(t_sinet));

save sine_data.dat t_sinet -ascii; % ascii marks text file, this is write mode

% read a file and load values
sines = load("sine_data.dat");
disp(sines)

% append cos values to the file above
coses = cos(2*pi*F*t);
save sine_data.dat coses -ascii -append;

file_data = load("sine_data.dat");
disp(file_data);