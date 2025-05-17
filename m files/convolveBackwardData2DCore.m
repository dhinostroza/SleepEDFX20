function dLossdX = convolveBackwardData2DCore( ...
    imageSize, W, dLossdZ, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth, ...
    dilationHeight, dilationWidth, ...
    groups)
% convolveBackwardData2DCore   Backpropagate through a
% convolutional layer to get the derivative with respect to the input.

% Copyright 2017-2018 The MathWorks, Inc.

if nargin < 11
    dilationHeight = 1;
    dilationWidth = 1;
end

if nargin < 12
    groups = [];
end

dLossdX = builtin('_batchconvBackwardData', imageSize, dLossdZ, W, ...
    [padTop padLeft padBottom padRight], [strideHeight strideWidth],...
    [dilationHeight dilationWidth], groups);

xend

