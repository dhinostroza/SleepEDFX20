classdef SpectrogramSequenceDatastore < matlab.io.Datastore & ...
                                        matlab.io.datastore.Shuffleable % REMOVED MiniBatchable
% SpectrogramSequenceDatastore Creates overlapping sequences from epoch data
% *** Simplified inheritance, removed MiniBatchable and progress method ***

    properties
        EpochSpectrograms % [NumEpochs x H x W x C] array
        EpochLabels       % [NumEpochs x 1] categorical array
        SequenceLength    % Length of sequences to generate
        MiniBatchSize     % Size of mini-batches to return (still used by read)
        NumSequences      % Total number of sequences
        SequenceIndices   % Indices defining the start of each sequence
        CurrentIndex      % Index for reading data
    end

    properties(SetAccess = protected)
        NumClasses        % Number of unique classes
        ClassNames        % Names of the classes
    end

    methods
        function ds = SpectrogramSequenceDatastore(spectrograms, labels, seqLength, miniBatchSize)
            % Constructor
            ds.EpochSpectrograms = spectrograms;
            ds.EpochLabels = categorical(labels);
            ds.SequenceLength = seqLength;
            ds.MiniBatchSize = miniBatchSize; % Store batch size
            numEpochs = size(ds.EpochSpectrograms, 1);
            if numEpochs < ds.SequenceLength, error('SpectrogramSequenceDatastore:NotEnoughData', 'Not enough epochs (%d) for sequence length %d.', numEpochs, ds.SequenceLength); end
            ds.NumSequences = numEpochs - ds.SequenceLength + 1;
            ds.SequenceIndices = (1:ds.NumSequences)';
            ds.ClassNames = categories(ds.EpochLabels);
            ds.NumClasses = numel(ds.ClassNames);
            ds = ds.reset();
        end

        function tf = hasdata(ds)
            % Check if there is more data to read
            tf = ds.CurrentIndex <= ds.NumSequences;
        end

        function [data, info] = read(ds)
            % Read one mini-batch of data
            if ~hasdata(ds), error('SpectrogramSequenceDatastore:NoMoreData', 'No more data to read.'); end
            numToEnd = ds.NumSequences - ds.CurrentIndex + 1;
            % Use the MiniBatchSize property we stored
            batchSize = min(ds.MiniBatchSize, numToEnd);
            indicesToRead = ds.CurrentIndex : (ds.CurrentIndex + batchSize - 1);
            startEpochIndices = ds.SequenceIndices(indicesToRead);
            batchSequences = cell(batchSize, 1);
            batchLabels = ds.EpochLabels(startEpochIndices + ds.SequenceLength - 1);
            H = size(ds.EpochSpectrograms, 2); W = size(ds.EpochSpectrograms, 3); C = size(ds.EpochSpectrograms, 4);
            for i = 1:batchSize
                startIdx = startEpochIndices(i); endIdx = startIdx + ds.SequenceLength - 1;
                seqData = ds.EpochSpectrograms(startIdx:endIdx, :, :, :);
                seqImagesCell = cell(1, ds.SequenceLength);
                 for t = 1:ds.SequenceLength, seqImagesCell{t} = squeeze(seqData(t, :, :, :)); end
                 batchSequences{i} = seqImagesCell;
            end
            % Return data as a table with variable names matching network inputs/outputs
            data = table(batchSequences, batchLabels);
            data.Properties.VariableNames = {'input', 'main_output'}; % Match network layer names

            ds.CurrentIndex = ds.CurrentIndex + batchSize;
            info.BatchSize = batchSize; % Include batch size info
        end

        function data = readall(ds)
            % Read all data from the datastore
            ds.reset(); dataCell = {}; infoCell = {};
            while ds.hasdata()
                [dataBatch, infoBatch] = ds.read(); dataCell{end+1} = dataBatch; infoCell{end+1} = infoBatch;
            end
            try data = vertcat(dataCell{:}); catch, warning('Could not vertically concatenate batches in readall.'); data = dataCell; end
             ds.reset();
        end

        function data = preview(ds)
            % Return the first data record
            ds.reset();
            if ds.hasdata()
                currentBatchSize = ds.MiniBatchSize; ds.MiniBatchSize = 1;
                [data, ~] = ds.read(); ds.MiniBatchSize = currentBatchSize; ds.reset();
            else, data = []; end
        end

        function ds = reset(ds)
            % Reset to the beginning of the data
            ds.CurrentIndex = 1;
        end

        function dsNew = shuffle(ds)
            % Shuffle the sequence indices
            dsNew = copy(ds); idx = randperm(dsNew.NumSequences);
            dsNew.SequenceIndices = dsNew.SequenceIndices(idx);
            dsNew = dsNew.reset();
        end

        % REMOVED progress method as MiniBatchable is removed

    end
end