function Y = pySlice(X, dim, startInd, endInd, step, removeDim)
%PYSLICE Slices a tensor from 'start' to 'end' in intervals of 'step'.
% at::Tensor at::slice(const at::Tensor &self, int64_t dim = 0, c10::optional<int64_t> start = c10::nullopt, c10::optional<int64_t> end = c10::nullopt, int64_t step = 1)

%   Copyright 2022-2023 The MathWorks, Inc.

import maglev_rnn_traced.ops.*

dim = dim.value;
startInd = startInd.value;
endInd = endInd.value;
step = step.value;

% If called for 'Select' operator, set endInd+1
if removeDim
    endInd = endInd + 1;
end

% Convert the input data to reverse-Python dimension order
Xval = X.value;
Xrank = X.rank;

% Slice the data
% Set default Axes and Steps if not supplied
if isempty(dim)
    dim = 0:Xrank-1;   % All axes
end
dim(dim<0) = dim(dim<0) + Xrank; % Handle negative Axes.
if isempty(step)
    step = ones(1, numel(startInd));
end
% Init all dims to :
S.subs = repmat({':'}, 1, Xrank);
S.type = '()';

%Convert dim to reverse-Python dimension
RevDim = Xrank - dim;

% Set Starts and Ends for each axis
for i = 1:numel(RevDim)
%     DLTDim = Xrank - dim(i);                                               % The DLT dim is the reverse of the ONNX dim.
    
    %In scripted models, startInd/endInd could be empty (optional). In such
    %cases we set startInd to 0 and endInd to number of elements in the
    %dimension
    if isempty(startInd)
        startInd(i) = 0;
    end

    if isempty(endInd)
        endInd(i) = size(Xval,RevDim(i));
    end
    % "If a negative value is passed for any of the start or end indices,
    % it represents number of elements before the end of that dimension."
    if startInd(i) < 0
        startInd(i) = size(Xval,RevDim(i)) + startInd(i);
    end
    if endInd(i) < 0
        endInd(i) = max(-1, size(Xval,RevDim(i)) + endInd(i));                        % The -1 case is when we're slicing backward and want to include 0.
    end
    
    % "If the value passed to start or end is larger than the n (the number
    % of elements in this dimension), it represents n."
    if startInd(i) > size(Xval,RevDim(i))
        startInd(i) = size(Xval,RevDim(i));
    end
    if endInd(i) > size(Xval,RevDim(i))
        endInd(i) = size(Xval,RevDim(i));
    end
     if step(i) > 0
        S.subs{RevDim(i)} = 1 + (startInd(i) : step(i) : endInd(i)-1);            % 1 + (Origin 0 indexing with end index excluded)
     else
         S.subs{RevDim(i)} = 1 + (startInd(i) : step(i) : endInd(i)+1);            % 1 + (Origin 0 indexing with end index excluded)
     end
end
Yval = subsref(Xval, S);

%Condition for select operation, remove selected dimension and reduce rank
if removeDim
    Yrank = Xrank - 1;
    YvalSize = [size(Yval), ones(1, Xrank - ndims(Yval))];
    if(all(YvalSize(RevDim) == 1))
        YvalSize(RevDim) = [];
        Yval = reshape(Yval,YvalSize);
    end
else
    Yrank = Xrank;
end


label = repelem('U',Yrank);
Yval = dlarray(Yval, label);


Y = struct('value', Yval, 'rank', Yrank);
end