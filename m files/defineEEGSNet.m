% defineEEGSNet.m
% Defines the EEGSNet architecture (Corrected v5)
% *** Returns both full graph and CNN feature extractor graph ***

function [lgraph, lgraph_cnn_feat] = defineEEGSNet() % Modified output
    % Defines the EEGSNet architecture based on Li et al. 2022, IJERPH, 19, 6322

    fprintf('--- Defining EEGSNet Architecture (Corrected v5) ---\n');

    % --- Configuration ---
    inputSize = [76 60 3]; numClasses = 5; dropoutProb = 0.5; numHiddenUnitsLSTM = 128;

    % --- Input Layer ---
    layers = [imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')];
    lgraph = layerGraph(layers);

    % --- Block 1 ---
    lgraph = addLayers(lgraph, convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'b1_conv1')); lgraph = addLayers(lgraph, geluLayer('Name', 'b1_gelu1'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'b1_conv2')); lgraph = addLayers(lgraph, geluLayer('Name', 'b1_gelu2'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'b1_conv3')); lgraph = addLayers(lgraph, geluLayer('Name', 'b1_gelu3'));
    lgraph = addLayers(lgraph, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'b1_pool')); lgraph = addLayers(lgraph, batchNormalizationLayer('Name', 'b1_bn')); lgraph = addLayers(lgraph, dropoutLayer(dropoutProb, 'Name', 'b1_dropout'));
    lgraph = connectLayers(lgraph, 'input', 'b1_conv1'); lgraph = connectLayers(lgraph, 'b1_conv1', 'b1_gelu1'); lgraph = connectLayers(lgraph, 'b1_gelu1', 'b1_conv2'); lgraph = connectLayers(lgraph, 'b1_conv2', 'b1_gelu2'); lgraph = connectLayers(lgraph, 'b1_gelu2', 'b1_conv3'); lgraph = connectLayers(lgraph, 'b1_conv3', 'b1_gelu3'); lgraph = connectLayers(lgraph, 'b1_gelu3', 'b1_pool'); lgraph = connectLayers(lgraph, 'b1_pool', 'b1_bn'); lgraph = connectLayers(lgraph, 'b1_bn', 'b1_dropout');

    % --- Block 2 ---
    lgraph = addLayers(lgraph, convolution2dLayer(1, 16, 'Stride', 1, 'Name', 'b2_conv1')); lgraph = addLayers(lgraph, geluLayer('Name', 'b2_gelu1'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'b2_conv2')); lgraph = addLayers(lgraph, geluLayer('Name', 'b2_gelu2'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'b2_conv3')); lgraph = addLayers(lgraph, geluLayer('Name', 'b2_gelu3'));
    lgraph = addLayers(lgraph, averagePooling2dLayer(3, 'Stride', 1, 'Padding', 'same', 'Name', 'b2_pool')); lgraph = addLayers(lgraph, batchNormalizationLayer('Name', 'b2_bn')); lgraph = addLayers(lgraph, dropoutLayer(dropoutProb, 'Name', 'b2_dropout'));
    lgraph = connectLayers(lgraph, 'input', 'b2_conv1'); lgraph = connectLayers(lgraph, 'b2_conv1', 'b2_gelu1'); lgraph = connectLayers(lgraph, 'b2_gelu1', 'b2_conv2'); lgraph = connectLayers(lgraph, 'b2_conv2', 'b2_gelu2'); lgraph = connectLayers(lgraph, 'b2_gelu2', 'b2_conv3'); lgraph = connectLayers(lgraph, 'b2_conv3', 'b2_gelu3'); lgraph = connectLayers(lgraph, 'b2_gelu3', 'b2_pool'); lgraph = connectLayers(lgraph, 'b2_pool', 'b2_bn'); lgraph = connectLayers(lgraph, 'b2_bn', 'b2_dropout');

    % --- Downsample Block 2 Output ---
    lgraph = addLayers(lgraph, convolution2dLayer(1, 32, 'Stride', 2, 'Name', 'b2_res_downsample')); lgraph = connectLayers(lgraph, 'b2_dropout', 'b2_res_downsample');
    % --- Residual Connection 1 ---
    lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'add1')); lgraph = connectLayers(lgraph, 'b1_dropout', 'add1/in1'); lgraph = connectLayers(lgraph, 'b2_res_downsample', 'add1/in2');

    % --- Block 3 ---
    lgraph = addLayers(lgraph, convolution2dLayer(1, 20, 'Stride', 1, 'Name', 'b3_conv1')); lgraph = addLayers(lgraph, geluLayer('Name', 'b3_gelu1'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'b3_conv2')); lgraph = addLayers(lgraph, geluLayer('Name', 'b3_gelu2'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'b3_conv3')); lgraph = addLayers(lgraph, geluLayer('Name', 'b3_gelu3'));
    lgraph = addLayers(lgraph, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'b3_pool')); lgraph = addLayers(lgraph, batchNormalizationLayer('Name', 'b3_bn')); lgraph = addLayers(lgraph, dropoutLayer(dropoutProb, 'Name', 'b3_dropout'));
    lgraph = connectLayers(lgraph, 'add1', 'b3_conv1'); lgraph = connectLayers(lgraph, 'b3_conv1', 'b3_gelu1'); lgraph = connectLayers(lgraph, 'b3_gelu1', 'b3_conv2'); lgraph = connectLayers(lgraph, 'b3_conv2', 'b3_gelu2'); lgraph = connectLayers(lgraph, 'b3_gelu2', 'b3_conv3'); lgraph = connectLayers(lgraph, 'b3_conv3', 'b3_gelu3'); lgraph = connectLayers(lgraph, 'b3_gelu3', 'b3_pool'); lgraph = connectLayers(lgraph, 'b3_pool', 'b3_bn'); lgraph = connectLayers(lgraph, 'b3_bn', 'b3_dropout');

    % --- Block 4 ---
    lgraph = addLayers(lgraph, convolution2dLayer(1, 20, 'Stride', 1, 'Name', 'b4_conv1')); lgraph = addLayers(lgraph, geluLayer('Name', 'b4_gelu1'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'b4_conv2')); lgraph = addLayers(lgraph, geluLayer('Name', 'b4_gelu2'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'b4_conv3')); lgraph = addLayers(lgraph, geluLayer('Name', 'b4_gelu3'));
    lgraph = addLayers(lgraph, averagePooling2dLayer(3, 'Stride', 1, 'Padding', 'same', 'Name', 'b4_pool')); lgraph = addLayers(lgraph, batchNormalizationLayer('Name', 'b4_bn')); lgraph = addLayers(lgraph, dropoutLayer(dropoutProb, 'Name', 'b4_dropout'));
    lgraph = connectLayers(lgraph, 'add1', 'b4_conv1'); lgraph = connectLayers(lgraph, 'b4_conv1', 'b4_gelu1'); lgraph = connectLayers(lgraph, 'b4_gelu1', 'b4_conv2'); lgraph = connectLayers(lgraph, 'b4_conv2', 'b4_gelu2'); lgraph = connectLayers(lgraph, 'b4_gelu2', 'b4_conv3'); lgraph = connectLayers(lgraph, 'b4_conv3', 'b4_gelu3'); lgraph = connectLayers(lgraph, 'b4_gelu3', 'b4_pool'); lgraph = connectLayers(lgraph, 'b4_pool', 'b4_bn'); lgraph = connectLayers(lgraph, 'b4_bn', 'b4_dropout');

    % --- Downsample Block 4 Output ---
    lgraph = addLayers(lgraph, convolution2dLayer(1, 64, 'Stride', 2, 'Name', 'b4_res_downsample')); lgraph = connectLayers(lgraph, 'b4_dropout', 'b4_res_downsample');
    % --- Residual Connection 2 ---
    lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'add2')); lgraph = connectLayers(lgraph, 'b3_dropout', 'add2/in1'); lgraph = connectLayers(lgraph, 'b4_res_downsample', 'add2/in2');

    % --- Block 5 ---
    lgraph = addLayers(lgraph, convolution2dLayer(1, 20, 'Stride', 1, 'Name', 'b5_conv1')); lgraph = addLayers(lgraph, geluLayer('Name', 'b5_gelu1'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'b5_conv2')); lgraph = addLayers(lgraph, geluLayer('Name', 'b5_gelu2'));
    lgraph = addLayers(lgraph, convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'b5_conv3')); lgraph = addLayers(lgraph, geluLayer('Name', 'b5_gelu3'));
    lgraph = addLayers(lgraph, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'b5_pool')); lgraph = addLayers(lgraph, batchNormalizationLayer('Name', 'b5_bn')); lgraph = addLayers(lgraph, dropoutLayer(dropoutProb, 'Name', 'b5_dropout'));
    lgraph = connectLayers(lgraph, 'add2', 'b5_conv1'); lgraph = connectLayers(lgraph, 'b5_conv1', 'b5_gelu1'); lgraph = connectLayers(lgraph, 'b5_gelu1', 'b5_conv2'); lgraph = connectLayers(lgraph, 'b5_conv2', 'b5_gelu2'); lgraph = connectLayers(lgraph, 'b5_gelu2', 'b5_conv3'); lgraph = connectLayers(lgraph, 'b5_conv3', 'b5_gelu3'); lgraph = connectLayers(lgraph, 'b5_gelu3', 'b5_pool'); lgraph = connectLayers(lgraph, 'b5_pool', 'b5_bn'); lgraph = connectLayers(lgraph, 'b5_bn', 'b5_dropout');

    % --- Global Average Pooling ---
    lgraph = addLayers(lgraph, globalAveragePooling2dLayer('Name', 'gap')); lgraph = connectLayers(lgraph, 'b5_dropout', 'gap');
    % --- Flatten Layer ---
    lgraph = addLayers(lgraph, flattenLayer('Name', 'flatten_gap')); lgraph = connectLayers(lgraph, 'gap', 'flatten_gap');

    % --- Create CNN Feature Extractor Graph (lgraph_cnn_feat) ---
    % Add temporary classification head for training the CNN part
    lgraph_cnn_feat = addLayers(lgraph, fullyConnectedLayer(numClasses, 'Name', 'temp_cnn_fc'));
    lgraph_cnn_feat = addLayers(lgraph_cnn_feat, softmaxLayer('Name', 'temp_cnn_softmax'));
    lgraph_cnn_feat = addLayers(lgraph_cnn_feat, classificationLayer('Name', 'temp_cnn_output'));
    lgraph_cnn_feat = connectLayers(lgraph_cnn_feat, 'flatten_gap', 'temp_cnn_fc');
    lgraph_cnn_feat = connectLayers(lgraph_cnn_feat, 'temp_cnn_fc', 'temp_cnn_softmax');
    lgraph_cnn_feat = connectLayers(lgraph_cnn_feat, 'temp_cnn_softmax', 'temp_cnn_output');
    fprintf('--- CNN Feature Extractor Graph Defined (lgraph_cnn_feat) ---\n');

    % --- Complete Full Graph (lgraph) by adding LSTM and final head ---
    lgraph = addLayers(lgraph, bilstmLayer(numHiddenUnitsLSTM, 'OutputMode', 'sequence', 'Name', 'bilstm1'));
    lgraph = addLayers(lgraph, bilstmLayer(numHiddenUnitsLSTM, 'OutputMode', 'last', 'Name', 'bilstm2'));
    lgraph = connectLayers(lgraph, 'flatten_gap', 'bilstm1'); lgraph = connectLayers(lgraph, 'bilstm1', 'bilstm2');
    lgraph = addLayers(lgraph, fullyConnectedLayer(numClasses, 'Name', 'main_fc'));
    lgraph = addLayers(lgraph, softmaxLayer('Name', 'main_softmax'));
    lgraph = addLayers(lgraph, classificationLayer('Name', 'main_output'));
    lgraph = connectLayers(lgraph, 'bilstm2', 'main_fc'); lgraph = connectLayers(lgraph, 'main_fc', 'main_softmax'); lgraph = connectLayers(lgraph, 'main_softmax', 'main_output');

    fprintf('--- Full EEGSNet Architecture Defined (lgraph) ---\n');

end % End of function defineEEGSNet