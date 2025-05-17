classdef GELUHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % GELUHostStrategy   Execution strategy for running GELU on the host
    
    %   Copyright 2022 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ~)
            Z = nnet.internal.cnnhost.geluForward(X);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X)
            dX = nnet.internal.cnnhost.geluBackward(Z, dZ, X);
            dW = [];
        end
    end
end