classdef Convolution2DHostStridedConvStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % Convolution2DHostStridedConvStrategy   Execution strategy for running the convolution on the host
    
    %   Copyright 2017-2021 The MathWorks, Inc.
    
    properties(Access=private)
       PaddingStrategy
    end
    
    methods
        function this = Convolution2DHostStridedConvStrategy(paddingValue)
            this.PaddingStrategy = nnet.internal.cnn.layer.padding.createPaddingStrategy(paddingValue);
        end
        
        function [Z, memory] = forward(this, X, ...
                weights, bias, ...
                paddingSize, stride, dilation)
            
            spatialDims = 1:2;
            X = this.PaddingStrategy.forwardPad(X,paddingSize,spatialDims);
            paddingSize = this.PaddingStrategy.remainingPaddingSize(paddingSize);
            
            topPad    = paddingSize(1);
            bottomPad = paddingSize(2);
            leftPad   = paddingSize(3);
            rightPad  = paddingSize(4);

            verticalStride     = stride(1); 
            horizontalStride   = stride(2);
            verticalDilation   = dilation(1);
            horizontalDilation = dilation(2);
            
            Z = nnet.internal.cnnhost.stridedConv( ...
                X, weights, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride, ...
                verticalDilation, horizontalDilation, bias);

            memory = [];
        end
        
        function [dX,dW] = backward( this, ...
                X, weights, dZ, ...
                paddingSize, stride, dilation)
            
            originalPaddingSize = paddingSize;
            spatialDims = 1:2;
            X = this.PaddingStrategy.forwardPad(X,paddingSize,spatialDims);
            paddingSize = this.PaddingStrategy.remainingPaddingSize(paddingSize);
            
            topPad    = paddingSize(1);
            bottomPad = paddingSize(2);
            leftPad   = paddingSize(3);
            rightPad  = paddingSize(4);

            verticalStride     = stride(1); 
            horizontalStride   = stride(2);
            verticalDilation   = dilation(1);
            horizontalDilation = dilation(2);

            needsWeightGradients = nargout > 1;
            dX = nnet.internal.cnnhost.convolveBackwardData2D( ...
                X, weights, dZ, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride, ...
                verticalDilation, horizontalDilation);
            if needsWeightGradients
                dW{1} = nnet.internal.cnnhost.convolveBackwardFilter2D( ...
                    X, weights, dZ, ...
                    topPad, leftPad, ...
                    bottomPad, rightPad, ...
                    verticalStride, horizontalStride, ...
                    verticalDilation, horizontalDilation);
                dW{2} = nnet.internal.cnnhost.convolveBackwardBias2D(dZ);
            end
            
            % Gradients of padding.
            dX = this.PaddingStrategy.backwardPad(dX,originalPaddingSize);
        end
    end
end
