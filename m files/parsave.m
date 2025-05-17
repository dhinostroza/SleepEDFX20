function parsave(fname, varargin)
%PARSAVE Helper function to save variables inside a parfor loop.
%   PARSAVE(FNAME, VARNAME1, VARNAME2, ...) saves the variables specified
%   by VARNAME1, VARNAME2, ... from the caller's workspace into the file
%   specified by FNAME.

    % Get variable names from input
    varNames = cell(1, nargin-1);
    for i = 1:(nargin-1)
        varNames{i} = inputname(i+1); % Get the name of the variable in the caller workspace
    end

    % Create a struct to hold the variables
    varsToSave = struct();
    for i = 1:length(varNames)
        varsToSave.(varNames{i}) = varargin{i}; % Assign the variable value to the struct field
    end

    % Save the struct
    save(fname, '-struct', 'varsToSave', '-v7.3'); % Save fields of the struct as individual variables
end