% this chapter explains data structures in matlab

% 1. cell arrays
%{ heterogeneous arrays of data of different types 
%}


carray1 = {"Sambit Mohapatra", 33, 5.9} % creating a cell array
size(carray1)

carray2 = {carray1;
            {"MSD", 34, 6.0}};

carray1(1)

% an empty cell array
cmat = cell(3, 3)
size(cmat)

% indexing - values are indexed using {}, returns value of referenced cell
carray2{1}  % 1st cell is 1x3, all 3 entries
carray2{2}{1}  % 2nd cell (1x3), 1st value -> name

% () indexing retruns a cell array object, actual data indexing then
% depends on the content of the returned cell array
carray3 = {"xyz", 33, 6.5, [1, 2, 3;
                            4, 5, 6;
                            7, 8, 9];
           "abc", 33, 6.5, [1, 2, 3;
                            4, 5, 6;
                            7, 8, 9]
                            }

cx = carray3(1, 4)  % this is a matrix
cx{:}  % this gets actual contents of the matrix


% ----------- structure type ------------------

% structure with 3 fields
mystruct1 = struct(name="sambit", ...
                    age=33, ...
                    height=5.9);
mystruct1

mystruct1.name  % field access

mystruct1.nation = "india";  % adding new field later, but inefficient
mystruct1

xs = rmfield(mystruct1, "age");  % removing a field, returns a new structure
xs  % new structure with field removed
mystruct1  % this is unchanged after the removal

isfield(mystruct1, "gender")  % check for field in struct

if isfield(mystruct1, "age")
    disp("Age is a field")
else
    disp("Age is not a field")
end

fnames = fieldnames(mystruct1)

% fieldnames can also be enterd dynamically
mystruct1.("height")  % dynamic field indexing

fname = input("Enter field to see -> ", "s");
if isfield(mystruct1, fname)
    mystruct1.(fname)
else
    disp("Given field is not in struct")
end