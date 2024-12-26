function [persistCount,localCount] = persistentVars(countInit)
%{persistentVars
%Function demonstrates persistent variables - same as static vars
%}

persistent pCount;

% initializing persistent variables
if isempty(pCount)
    pCount = countInit;
end

pCount = pCount + 1;
localCount = countInit + 1;
persistCount = pCount;

end