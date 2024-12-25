%{
This chapter is about loops and vectorizing code
%}

% for loop to print multiplication table of N
N = 5;
for i=1:10
    fprintf("%d * %i = %d\n", N, i, N*i);
end

% for loop with a step
for i = 1 : 5 : 50
    fprintf("%d * %i = %d\n", N, i, N*i);
end


% for loop reverse indexed
disp("For loop with reverse indices")
for i = 10: -1: 0
    fprintf("%d * %i = %d\n", N, i, N*i);
end

% for loop to print running sum of user inputs
sum = 0
for i = 1:5
    n = input("Enter a number -> ");
    sum = sum + n;
    fprintf("Sum = %d\n", sum);
end

% inputs to vector
% this is an inefficient method of extending vectors, 
vx = [];
for i = 1:5
    num = input("Enter a number -> ");
    vx = [vx, num];
    disp(vx);
    disp(size(vx))
end

% subplots with for loop
for i = 1:2
    x=linspace(0,2*pi,20*i);
    y=sin(x);
    subplot(1,2,i)
    plot(x,y,'ko')
    xlabel('x')
    ylabel('sin(x)')
    title('sin plot')
end

% nested for loops
for i = 5:-1:1
    for j = 1:i
        fprintf("*\t");
    end
    fprintf("\n");
end
