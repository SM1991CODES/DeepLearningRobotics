function [outputArg1] = getCumSum(inputArg1)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

outputArg1 = [];
sm = 1;
for i = 1 : inputArg1
    sm = sm + i;
    outputArg1 = [outputArg1, sm];
end