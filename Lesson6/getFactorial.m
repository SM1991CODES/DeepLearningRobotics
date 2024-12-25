function [outputArg1] = getFactorial(inputArg1)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

outputArg1 = 1;
for i = 1 : inputArg1
    outputArg1 = outputArg1 * i;
end