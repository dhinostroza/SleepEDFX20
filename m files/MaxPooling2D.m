classdef MaxPooling2D < nnet.internal.cnn.layer.FunctionalLayer ...
    & nnet.internal.cnn.layer.CPUFusablePoolingLayer
    % MaxPooling2D   2-D Max pooling layer implementation
    
    %   Copyright 2015-2024 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % PaddingSize   The padding applied to the input along the edges
        %   The padding that is applied along the edges. This is a row
        %   vector [t b l r] where t is the padding to the top, b is the
        %   padding applied to the bottom, l is the padding applied to the
        %   left, and r is the padding applied to the right.
        PaddingSize

        % PoolOverTime   True for layers performing temporal pooling.
        PoolOverTime
        
        % DistributeOverTime   True for layers distributing the operation
        % over time
        DistributeOverTime
        
        % Index of the dimensions to be pooled
        PoolingDimIdx
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'maxpool'
    end
    
    properties (SetAccess = private)
        % InputNames   The layer has one input
        InputNames = {'in'}
        
        % HasSizeDetermined   Specifies if all size parameters are set
        %   Required to be false so that second output can be configured.
        HasSizeDetermined = false;
        
        % PoolSize   The height and width of a pooling region
        %   The size the pooling regions. This is a vector [h w] where h is
        %   the height of a pooling region, and w is the width of a pooling
        %   region.
        PoolSize
        
        % PaddingMode   The mode used to determine the padding
        %   The mode used to calculate the PaddingSize property. This can
        %   be:
        %       'manual'    - PaddingSize is specified manually.
        %       'same'      - PaddingSize will be calculated so that the
        %                     output size is the same size as the input
        %                     when the stride is 1. More generally, the
        %                     output size will be ceil(inputSize/stride),
        %                     where inputSize is the height and width of
        %                     the input.
        PaddingMode
        
        % HasUnpoolingOutputs   Specifies whether this layer should have extra
        %   outputs that can be used for unpooling. This can be:
        %       true      - The layer will have two extra outputs 'indices'
        %                   and 'size' which can be used with a max
        %                   unpooling layer.
        %       false     - The layer will have one output 'out'.
        HasUnpoolingOutputs

        % ExecutionStrategyFactory Factory to create the execution strategy
        % depending on the type of environment for which it should be run.
        % See setHostStrategy.
        ExecutionStrategyFactory

        % ExecutionStrategy Captures the implementation for how to perform
        % the predict behavior for a given layer depending on the
        % environment and additional parameters.
        ExecutionStrategy
    end
    
    properties
        % Learnables   Empty
        Learnables
    end
    
    properties(SetAccess=protected)
        % LearnablesName   Empty
        LearnablesNames        
    end
    
    properties (SetAccess = protected)
        % IsInFunctionalMode   Returns true if layer is currently being
        % used in "functional" mode (i.e. in dlnetwork). Required by
        % FunctionalLayer interface. On construction, all layers are set up
        % for usage in DAGNetwork.
        IsInFunctionalMode = false
    end
    
    properties(SetAccess = private, Dependent)
        % OutputNames   The number of outputs depends on whether
        %               'HasUnpoolingOutputs' is true or false
        OutputNames
    end
    
    methods
        function this = MaxPooling2D( ...
                name, poolSize, stride,  paddingMode, paddingSize, unpoolingOutputs)
            this.Name = name;
            
            % Set hyperparameters
            this.PoolSize = poolSize;
            this.Stride = stride;
            this.PaddingMode = paddingMode;
            this.PaddingSize = paddingSize;
            if nargin == 6
                this.HasUnpoolingOutputs = unpoolingOutputs;
            else
                this.HasUnpoolingOutputs = false;
            end
            
            this.ExecutionStrategyFactory = nnet.internal.cnn.layer.strategy.MaxPooling2DStrategyFactory();
            this = selectHostStrategy(this);
        end
        
        function Z = predict(this, X)
            % Find the input size
            inputSize = this.getInputSizeAlongPoolingDimensions(X);

            % Select stride for fused operations
            stride = this.selectStride();

            % Find the padding
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, this.PoolSize, ...
                stride, inputSize);

            % Padding is stored as [top bottom left right] but the function
            % expects [top left; bottom right]
            paddingSize = [paddingSize(1), paddingSize(3); 
                            paddingSize(2), paddingSize(4)];

            % Call the forward method of the execution strategy
            Z = this.ExecutionStrategy.forward( ...
                X, ...
                this.PoolSize, ...
                paddingSize, ...
                stride);
        end
        
        function [dX,dW] = backward(this, X, Z, dZ, ~)
            % Find the input size. Here, we don't need to recompute
            % PoolingDimIdx as 'backward' does not get called in
            % functional mode.
            inputSize = size(X, this.PoolingDimIdx);

            % Find the padding
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, this.PoolSize, ...
                this.Stride, inputSize);

            % Padding is stored as [top bottom left right] but the function
            % expects [top left; bottom right]
            paddingSize = [paddingSize(1), paddingSize(3); 
                            paddingSize(2), paddingSize(4)];

            % Call the backward method of the execution strategy
            [dX,dW] = this.ExecutionStrategy.backward( ...
                Z, dZ, X, ...
                this.PoolSize, ...
                paddingSize, ...
                this.Stride);
        end

        function this = configureForInputs(this, Xs)
            X = Xs{1};

            hasTimeDimension = ~isempty(finddim(X, 'T'));
            numSpatialDimensions  = nnz(dims(X)=='S');

            % Set PoolOverTime, DistributeOverTime, and PoolingDimIdx 
            % properties
            this.assertValidNumPoolingDimensions(hasTimeDimension, numSpatialDimensions);
            [this.PoolOverTime, this.DistributeOverTime] = iSetSpaceTimeProperties(hasTimeDimension, ...
                numSpatialDimensions);
            this.PoolingDimIdx = iFindPoolingDimIdx(X, this.PoolOverTime);

            % For DAGNetwork, data must have a B dimensions and cannot have 
            % U dimensions.
            if ~this.IsInFunctionalMode
                this.assertInputHasBatchDim(X);
                this.assertInputHasNoUndefinedDims(X);

                % The temporal built-in expects a channel dimension, so we
                % check for one here
                if this.PoolOverTime
                    numChannels = getSizeForDims(X,'C');
                    this.assertInputHasChannelDim(numChannels);
                end
            end

            if iIsTheStringSame(this.PaddingMode)
                this.PaddingSize = iCalculateSamePadding( ...
                    this.PoolSize, this.Stride, size(X,this.PoolingDimIdx));
            end

            % Check that the pooling window is smaller than the input size
            this.assertPoolSizeSmallerThanInput(X);

            if this.HasUnpoolingOutputs
                % Unpooling outputs for input with time dim only supported
                % when in functional mode and not pooling over time
                if this.IsInFunctionalMode
                    this.assertNotPoolOverTime()
                else
                    this.assertInputHasNoTimeDim(hasTimeDimension)
                end

                % Overlapping pooling windows are not supported when
                % output indices are requested
                this.assertNoOverlapBetweenPoolingRegions()
            end
        end

        function Zs = forwardExampleInputs(this,Xs)
            X = Xs{1};

            hasTimeDimension = ~isempty(finddim(X, 'T'));
            numSpatialDimensions  = nnz(dims(X)=='S');
            this.assertValidNumPoolingDimensions(hasTimeDimension, numSpatialDimensions);

            % For the DAGNetwork case, we check the data has a B dimension
            % and no U dimensions. In dlnetwork, this isn't a problem
            % because maxpool doesn't need a B dimension and will ignore
            % any U dimensions
            if ~this.IsInFunctionalMode
                this.assertInputHasBatchDim(X);
                this.assertInputHasNoUndefinedDims(X);

                % The temporal built-in expects a channel dimension, so we
                % check for one here
                if this.PoolOverTime
                    numChannels = getSizeForDims(X,'C');
                    this.assertInputHasChannelDim(numChannels);
                end
            end

            % Check that the pooling window is smaller than the input size
            this.assertPoolSizeSmallerThanInput(X);

            % Prepare output array
            Z = X;

            % Determine the size of the pooling dimension after pooling.
            inputSize = this.getInputSizeAlongPoolingDimensions(X);
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, this.PoolSize, ...
                this.Stride, inputSize);

            heightAndWidthPadding = iCalculateHeightAndWidthPadding(paddingSize);
            outputSize = floor((inputSize + ...
                heightAndWidthPadding - this.PoolSize)./this.Stride) + 1;
            if this.PoolOverTime
                % Pooling dimensions are ST
                Z = setSizeForDim(Z,'S',outputSize(1));
                Z = setSizeForDim(Z,'T',outputSize(2));
            else
                % Pooling dimensions are SS
                Z = setSizeForDim(Z,'S',outputSize);
            end
            
            % Max pooling output
            Zs{1} = Z;
            if this.HasUnpoolingOutputs
                % Unpooling outputs for input with time dim only supported
                % when in functional mode and not pooling over time
                if this.IsInFunctionalMode
                    this.assertNotPoolOverTime()
                else
                    this.assertInputHasNoTimeDim(hasTimeDimension)
                end

                % Overlapping pooling windows are not supported when
                % output indices are requested
                this.assertNoOverlapBetweenPoolingRegions()
                
                Zs2Format = 'SSCB';
                Zs3Format = 'SSSSCB';
                % Dlnetworks can have 'SSC' data, so need to remove batch 
                % dimension for these cases
                if this.IsInFunctionalMode && isempty(getSizeForDims(X,'B'))
                    Zs2Format(end) = [];
                    Zs3Format(end) = [];
                end

                % Linear indices of pooling outputs
                Zs{2} = iMakeSizeOnlyArray([prod([outputSize getSizeForDims(X,'C')]) 1 1 getSizeForDims(X,'B')], Zs2Format);
                
                % Size of spatial dimensions for max unpooling output
                Zs{3} = iMakeSizeOnlyArray([1 4 1 getSizeForDims(X,'S') getSizeForDims(X,'B')], Zs3Format);
            end
        end
        
        function this = initializeLearnableParameters(this, ~)
        end
        
        function this = setQuantizationInfo(this, quantizationInfo)
            this = setQuantizationInfo@nnet.internal.cnn.layer.Layer(this, quantizationInfo);
            
            % Create a new floating point Factory and set the current
            % EnvironmentType on the factory
            floatingPointFactory = nnet.internal.cnn.layer.strategy.MaxPooling2DStrategyFactory();
            floatingPointFactory.EnvironmentType = this.ExecutionStrategyFactory.EnvironmentType;
            
            if ~isempty(quantizationInfo) && quantizationInfo.isQuantizationEnabled
                this.ExecutionStrategyFactory =  deep.internal.quantization.executionstrategyfactory.MaxPooling2DStrategyFactory(floatingPointFactory);
                this.ExecutionStrategy = this.ExecutionStrategyFactory.makeDecorated(this);
            else
                this.ExecutionStrategyFactory = floatingPointFactory;
            end
        end

        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
            this = selectHostStrategy(this);
        end
        
        function this = setupForGPUPrediction(this)
            this = selectGPUStrategy(this);
        end
        
        function this = setupForHostTraining(this)
            this = selectHostStrategy(this);
        end
        
        function this = setupForGPUTraining(this)
            this = selectGPUStrategy(this);
        end
        
        function val = get.OutputNames(this)
            if this.HasUnpoolingOutputs
                val = {'out','indices','size'};
            else
                val = {'out'};
            end
        end

        function inputSize = getInputSizeAlongPoolingDimensions(this,X)
            % Compute the size of X along the pooling dimension.
            % When the layer IsInFunctionalMode the labeled dimensions of X
            % can vary on each call, so recompute the indices of
            % the pooling dimension.
            if this.IsInFunctionalMode
                idx = iFindPoolingDimIdx(X,this.PoolOverTime);
            else
                idx = this.PoolingDimIdx;
            end
            inputSize = size(X,idx);
        end
    end
    
    methods(Access = private)
        function assertValidNumPoolingDimensions(~, hasTimeDimension, numSpatialDimensions)
            % Data must have either:
            % a) Two spatial dimensions (SSCB)
            % b) One spatial dimensions and one temporal dimension (SCBT)
            % c) Two spatial dimensions and one temporal dimension (SSCBT)
            hasTwoContinuousDims = (hasTimeDimension + numSpatialDimensions == 2);
            isSpatioTemporal = hasTimeDimension && (numSpatialDimensions == 2);
            if ~(hasTwoContinuousDims || isSpatioTemporal)
                error( message('nnet_cnn:internal:cnn:layer:MaxPooling2D:InvalidNumSpatioTemporalDims', ...
                    numSpatialDimensions, double(hasTimeDimension)));
            end
        end

        function assertInputHasChannelDim(~, numChannels)
            % Assert that the input data has a channel dimension
            if isempty(numChannels)
                error(message('nnet_cnn:internal:cnn:layer:MaxPooling2D:MissingChannelDimension'))
            end
        end

        function assertInputHasBatchDim(~, X)
            % Assert that the input data has a batch dimension
            if isempty(finddim(X,'B'))
                error(message('nnet_cnn:internal:cnn:layer:MaxPooling2D:MissingBatchDimension'))
            end
        end
                
        function assertInputHasNoUndefinedDims(~, X)
            % Assert that the input data has no undefined dimensions
            if any(finddim(X,'U'))
                error(message('nnet_cnn:internal:cnn:layer:MaxPooling2D:InputHasUndefinedDimensions'))
            end
        end

        function assertInputHasNoTimeDim(~, hasTimeDimension)
            % Assert that the input data has no temporal dimension
            if hasTimeDimension
                error(message('nnet_cnn:internal:cnn:layer:MaxPooling2D:NetworkWithUnpoolingOutputsHasTemporalDimensions'))
            end
        end

        function assertNotPoolOverTime(this)
            if this.PoolOverTime
                error(message('nnet_cnn:internal:cnn:layer:MaxPooling2D:NetworkWithUnpoolingOutputsIsPoolOverTime'))
            end   
        end

        function assertNoOverlapBetweenPoolingRegions(this)
            poolingRegionsOverlap = any(this.Stride < this.PoolSize);
            
            if poolingRegionsOverlap
                error(message('nnet_cnn:layer:MaxPooling2DLayer:IndicesRequireNonOverlappingPoolingRegion'));
            end
        end

        function assertPoolSizeSmallerThanInput(this, X)
            inputSize = this.getInputSizeAlongPoolingDimensions(X);
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, this.PoolSize, ...
                this.Stride, inputSize);
            heightWidthAndDepthPadding = iCalculateHeightAndWidthPadding(paddingSize);
            effectiveInputSize = inputSize + heightWidthAndDepthPadding;
            poolSizeSmallerThanInput = all(this.PoolSize <= effectiveInputSize);

            if ~poolSizeSmallerThanInput
                error( message('nnet_cnn:internal:cnn:layer:MaxPooling2D:PoolSizeLargerThanInput') );
            end
        end
        
        function this = selectGPUStrategy(this)
            % Update layer execution strategy
            this.ExecutionStrategyFactory.EnvironmentType = 'GPU';
            this.ExecutionStrategy = this.ExecutionStrategyFactory.makeStrategy(this);
        end
        
        function this = selectHostStrategy(this)
            % Update layer execution strategy
            this.ExecutionStrategyFactory.EnvironmentType = 'Host';
            this.ExecutionStrategy = this.ExecutionStrategyFactory.makeStrategy(this);
        end
    end
    
    methods(Access=protected)
        function this = setFunctionalStrategy(this)
            % Update layer execution strategy
            this.ExecutionStrategyFactory.EnvironmentType = 'Functional';
            this.ExecutionStrategy = this.ExecutionStrategyFactory.makeStrategy(this);
        end

        function [callsign, option] = getLayerSpecificArguments(layer)
            % getLayerSpecificArguments  Returns the call sign of the layer and
            % the optional algorithmic switch.

            callsign = 'maxpool';
            % 2-D max pooling layers can return indices.
            option = layer.HasUnpoolingOutputs;
        end
    end
    
    methods (Hidden)
        function attr = modifyDefaultFusionAttributes(layer, attr, ~)
            attr = setAttributesForMovingWindowFunctions(attr, ...
                layer.PoolOverTime, layer.PaddingMode);
        end
    end
end

function [poolOverTime, distributeOverTime] = iSetSpaceTimeProperties(hasTimeDimension, numSpatialDimensions)
layerDimension = 2;
[poolOverTime, distributeOverTime] = nnet.internal.cnn.layer.util.getSpaceTimePropertiesForLayer(hasTimeDimension, ...
    numSpatialDimensions, layerDimension);
end

function poolingDimIdx = iFindPoolingDimIdx(X, poolOverTime)
if poolOverTime
    numSpatialDims = 1;
else
    numSpatialDims = 2;
end
poolingDimIdx = nnet.internal.cnn.layer.util.findOperationDimIdx(X, poolOverTime, numSpatialDims);
end

function paddingSize = iCalculatePaddingSizeFromInputSize( ...
    paddingMode, paddingSize, filterOrPoolSize, stride, inputSize)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSizeFromInputSize( ...
    paddingMode, paddingSize, filterOrPoolSize, stride, inputSize);
end

function heightAndWidthPadding = iCalculateHeightAndWidthPadding(paddingSize)
heightAndWidthPadding = nnet.internal.cnn.layer.padding.calculateHeightAndWidthPadding(paddingSize);
end

function tf = iIsTheStringSame(x)
tf = nnet.internal.cnn.layer.padding.isTheStringSame(x);
end

function paddingSize = iCalculateSamePadding(poolSize, stride, inputSize)
paddingSize = nnet.internal.cnn.layer.padding.calculateSamePadding(poolSize, stride, inputSize);
end

function dlX = iMakeSizeOnlyArray(varargin)
dlX = deep.internal.PlaceholderArray(varargin{:});
end
