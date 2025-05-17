classdef BatchNormalizationHostStrategy
    % BatchNormalizationHostStrategy   Execution strategy for running batch normalization on the host
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forwardTrain(~, X, beta, gamma, epsilon, channelDim)
            [Z,batchMean,invSqrtVarPlusEps] = ...
                nnet.internal.cnnhost.batchNormalizationForwardTrain(X, beta, gamma, epsilon, channelDim);
            memory = {batchMean, invSqrtVarPlusEps};
        end
        
        function Z = forwardPredict(~, X, beta, gamma, epsilon, inputMean, inputVar, channelDim)
            Z = nnet.internal.cnnhost.batchNormalizationForwardPredict(X, beta, gamma, epsilon, inputMean, inputVar, channelDim);
        end
        
        function [dX,dW] = backward(~, ~, dZ, X, gamma, epsilon, memory, channelDim)
            [batchMean, invSqrtVarPlusEps] = deal(memory{:});
            args = { dZ, X, gamma, epsilon, batchMean, invSqrtVarPlusEps, channelDim };
            needsWeightGradients = nargout > 1;
            if ~needsWeightGradients
                dX = nnet.internal.cnnhost.batchNormalizationBackward( args{:} );
            else
                [dX,dW{1},dW{2}] = nnet.internal.cnnhost.batchNormalizationBackward( args{:} );
            end
        end
        
    end
end
