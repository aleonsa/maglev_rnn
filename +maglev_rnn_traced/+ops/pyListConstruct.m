function outStructs = pyListConstruct(varargin)
%PYLISTCONSTRUCT Groups inputs tensors into a list of tensors

%   Copyright 2022 The MathWorks, Inc.

import maglev_rnn_traced.ops.*

outStructs = [varargin{:}];
end

