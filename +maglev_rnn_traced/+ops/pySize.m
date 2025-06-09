function Yval = pySize(X, dim)
%PYSIZE Returns the size of the input tensor at the given dim
% If dim is empty, return the sizes of all dimensions
% int64_t at::size(const Tensor &tensor, int64_t dim)

%   Copyright 2022-2023 The MathWorks, Inc.

import maglev_rnn_traced.ops.*

dim = dim.value;

Xval = X.value;
Xrank = X.rank;

if Xrank == 0 % X is a scalar
    Yval = [];
elseif Xrank == 1 % X is a vector
    Yval = size(Xval);
else
    % Convert dim to reverse-pytorch order
    if isempty(dim)
        dim = 0:Xrank-1;
    end

    % Convert negative indices to positive indices
    dim(dim < 0) = dim(dim < 0) + Xrank;

    % Convert forward- to reverse- dimension order
    dltDim = Xrank - dim;    

    % Get size
    Yval = size(Xval, dltDim);    
end

% Set the output rank
if numel(dltDim) == 1
    Yrank = 0;
else
    Yrank = 1;
end

Yval = struct('value', int64(Yval(:)), 'rank', Yrank);
end