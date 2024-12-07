%{
THis chapter deals with scripts, input output and custom logic
%}

% inputs can be numbers, vectors, strings 
% a single number in matlab is a 1x1 vector
name = input("Enter your name: ", "s");  % "s" means character array input
name_s = string(name);  % character array needs explicit casting to string
age = input("Enter your age:");
height = input("Enter your height in feets: ");

% %.2f -> 2 values after decimal point
fprintf("Hello %s, you are %d years old and %.2f feet tall\n", name, age, height);

% when printig vectors and matrices, uses disp
m1 = randn(5); 
disp(m1);
fprintf("%.3f\n", m1);  % this will print one value at a time and will flatten the matrix column wise

%------- take radius as input and print area of circle -----------%

r = input("Enter radius -> ");
area = pi * (r^2);
fprintf("Area of a circle with radius = %f = %f\n", r, area);