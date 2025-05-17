function dLossdW = convolveBackwardFilter2D(X, W, dLossdZ, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth, ...
    dilationHeight, dilationWidth, ...
    groups)
% convolveBackwardFilter2D   Backpropagate through a convolutional layer to
% get the derivative with respect to the filters.
%
% Inputs:
% X - The input to the convolutional layer. An (H)x(W)x(C)x(N) array.
% W - The filters for the convolutional layer. We only pass these so that
%     we can get their dimensions. An (R)x(S)x(C)x(K) array.
%     C is the number of channels per group, K is the number of filters.
% dLossdZ - The derivative of the loss with respect to the output of the
% convolutional layer. Therefore the array size is
% floor((H + 2*padHeight - effectiveR)/strideHeight) + 1 x
% floor((W + 2*padWidth - effectiveS)/strideWidth) + 1 x
% (K) x (N) ,
% where effectiveR = (R-1) * dilationHeight + 1,
% effectiveS = (S-1) * dilationWidth + 1.
% padTop - Padding on the top.
% padLeft - Padding on the left.
% padBottom - Padding on the bottom.
% padRight - Padding on the right.
% strideHeight - The stride in the y direction.
% strideWidth - The stride in the x direction.
% dilationHeight - The dilation in the y direction. Default = 1.
% dilationWidth - The dilation in the x direction. Default = 1.
% groups - The filter groups. Default = [].
%
% Output:
% dLossdW - The derivative of the loss with respect to the filters. An
% (R)x(S)x(C)x(K) array.

%   Copyright 2016-2018 The MathWorks, Inc.

% The height and width of the filters. Note that this cannot be deduced
% from dLossdZ and X.

if nargin < 11
    dilationHeight = 1;
    dilationWidth = 1;
end

if nargin < 12
    groups = [];
end

dLossdW = builtin('_batchconvBackwardFilter', [size(W,1), size(W,2)], ...
    dLossdZ, X, [padTop padLeft padBottom padRight], ...
    [strideHeight strideWidth], [dilationHeight dilationWidth], ...
    groups);
end
