%{
This demonstrates functions in matlab.
Function name should be same as the file in which the function is.

General Syntax:
function <output args> = <func. name> (<input args.>)
...

...
end

- Function variables have local scope limited within the function.
- Script and command window share the same workspace => all variables in
scripts are accesible in the command window, but, not function variables

A script can have a local function defined in it at the end. This function
is only accessible to that script
%}

disp(calcArea([1, 2, 3])); % calling a function


[v1, v2, v3] = mathOps(5, 9);
disp("Sum:");
disp(v(1));

% finding correlation between 2 vectors
a1 = 1:10;
b1 = a1 .^ 2;
corrcoef(a1, b1)

a1 = randn(1, 10);
b1 = randn(1, 10);
corrcoef(a1, b1)


function [sum, diff, prod] = mathOps(op1, op2)
%{
Author: Sambit Mohapatra
Date: 08/12/24

This is a local function taking 2 arguments and returning 3
%}
sum = op1 + op2;
diff = op1 - op2;
prod = op1 .* op2;

end