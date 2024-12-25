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

