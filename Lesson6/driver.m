% this demonstrates use of modular programming
%{
Each function is saved as a separate matlab function file.
A driving matlab script then calls them
%}

in_data = getUserInput();
disp(in_data);

fprintf("factorial of %d = %d\n", in_data, getFactorial(in_data));

disp(getCumSum(in_data));
%fprintf("cumsum 1  -> %d = %d\n", in_data, getCumSum(in_data));

p_l = persistentVars(10);
disp(p_l)
p_l = persistentVars(10);
disp(p_l);
p_l = persistentVars(10);
disp(p_l);

disp(num2str(15.56));  % converts number to string

disp(str2double("15.56"));  % converting 
