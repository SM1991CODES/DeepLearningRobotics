% switch-case statements
%{
switch switch_expression
case caseexp1
action1
case caseexp2
action2
case caseexp3
action3
% etc: there can be many of these
otherwise
actionother
end

switch_expression must return an int type
The first matching case block is executed and then the block is exited
%}

score = 32;

% puts up a menu, returns index of the menu item choose
%score = menu("Enter student grade", ["10", "20", "30"]);

switch score

    % the switch_expr is compared to each case_expr, only exact matches are
    % executed
    case 10
        grade = 0;
    case 20
        grade = 1;
    case 30
        grade = 2;
    case {31, 32, 33, 34} % multiple case values to match any one
        grade = 9;
    otherwise
        grade = -1;
end
disp(grade);

% the is - functions

x = input("Enter a number");
y = input("Enter a number", "s");
isnumeric(x)
ischar(y)
isletter(x)

% isa(<var>, <type>) -> checking is param 1 is of a particular type
isa(12, "uint8");