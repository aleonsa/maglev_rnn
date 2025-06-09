function [Ystruct] = pyZeros(sizeInput)
%PYZEROS creates a tensor of zeroes of size (first argument)

%Copyright 2022 The MathWorks, Inc.

import maglev_rnn_traced.ops.*

sizeVal = [sizeInput.value];

revPytorchSize = fliplr(sizeVal);
Yval = zeros(revPytorchSize);
Yrank = numel(sizeVal);

Ystruct = makeStructForConstant(Yval,Yrank,"Tensor");

end
