%parallel.internal.parfor.ParforEngine
% Default PARFOR engine for (C++) process pools

% Copyright 2020-2023 The MathWorks, Inc.

classdef ParforEngine < parallel.internal.parfor.Engine
    
    properties (GetAccess = private, Constant)
        % Avoid building empty exceptions multiple times.
        EMPTY_EXCEPTION = MException.empty();
    end
    
    properties (GetAccess = private, SetAccess = immutable)
        Pool
        Session
        ParforF % parallel_function 'F' argument.
        MaxNumWorkers (1,1) int32
        InitData % In case we need to restart
    end
    
    properties (Access = private)
        ParforController
        
        NumWorkers (1,1) double
        
        AllIntervalsAdded (1,1) logical = false
        NormalCompletion  (1,1) logical = false
        AddedFilesToPool  (1,1) logical = false
        
        % Cell array of intervals that have been submitted but are not yet
        % complete. Indexed by the 'tag' produced by parallel_function - so
        % we rely on those tags being integers starting at 1.
        IncompleteIntervalsCell
        
        % Track the number of intervals that have been submitted to the controller,
        % but not yet collected.
        NumIntervalsInController (1,1) double = 0

        % Polling periods in milliseconds
        AwaitIntervalPollPeriod (1,1) double
        AwaitCompletedPollPeriod (1,1) double
    end

    methods (Static, Hidden)
        function [outAwaitIntervalPollPeriod, outAwaitCompletedPollPeriod] = ...
                getSetPollingPeriods(awaitIntervalPollPeriod, awaitCompletedPollPeriod)
            arguments
                awaitIntervalPollPeriod (1,1) double = 1000;
                awaitCompletedPollPeriod (1,1) double = 200;
            end
            persistent AWAIT_INTERVAL AWAIT_COMPLETED
            if isempty(AWAIT_INTERVAL)
                AWAIT_INTERVAL = awaitIntervalPollPeriod;
                AWAIT_COMPLETED = awaitCompletedPollPeriod;
            end
            outAwaitIntervalPollPeriod = AWAIT_INTERVAL;
            outAwaitCompletedPollPeriod = AWAIT_COMPLETED;
            if nargin > 0
                AWAIT_INTERVAL = awaitIntervalPollPeriod;
                AWAIT_COMPLETED = awaitCompletedPollPeriod;
            end
        end
    end
    
    methods
        function obj = ParforEngine(partitionMethod, partitionSize, ...
                pool, maxNumWorkers, initData, parforF)
            obj@parallel.internal.parfor.Engine(partitionMethod, partitionSize);
            
            iErrorCheckPool(pool);
            obj.Pool                = pool;
            obj.ParforF             = parforF;
            obj.MaxNumWorkers       = maxNumWorkers;
            obj.InitData            = initData;
            obj.Session             = pool.hGetSession;
            [obj.AwaitIntervalPollPeriod, obj.AwaitCompletedPollPeriod] = ...
                parallel.internal.parfor.ParforEngine.getSetPollingPeriods();

            feval('_pct_parforLog');
            
            obj.buildParforController()
        end

        
        function buildParforController(obj)
            obj.ParforController = []; % Ensure any old parfor controller is destroyed first
            
            try
                p = obj.Session.createParforController(obj.maxBacklog(), obj.MaxNumWorkers);
            catch err
                if ~obj.isSessionValidAndRunning()
                    errorMessageInput = iGetParpoolLinkForError();
                    error(message('parallel:lang:parfor:NoRunningSession', errorMessageInput));
                else
                    rethrow(err)
                end
            end
            
            obj.ParforController = p;
            obj.NumWorkers = p.getNumWorkers();
            
            obj.NumIntervalsInController = 0;

            [listener, getListFcn] = ...
                    parallel.internal.general.SerializationNotifier.createAccumulatingListener(); %#ok<ASGLU> 

            % Set the serialization context, so we can correctly transfer
            % special data types if required (e.g. Constant).
            contextGuard = parallel.internal.pool.SerializationContextGuard(obj.Session.UUID); %#ok<NASGU>
            
            obj.ParforController.beginLoop(parallel.internal.pool.optionallySerialize(obj.InitData));

            savedClasses = getListFcn();
            if ~isempty(savedClasses) && any(ismember({'distributed', 'Composite'}, savedClasses))
                error(message('parallel:lang:parfor:IllegalComposite'));
            end
        end
        
        function rebuildParforController(obj)
            % rebuildParforController - dispose of old parfor controller, build a new one, and submit any
            % intervals.
            if ~isempty(obj.ParforController)
                obj.ParforController.interrupt();
            end
            try
                obj.buildParforController();
            catch E
                % Can't rebuild for some reason - maybe pool has completely
                % crashed.
                if ~obj.isSessionValidAndRunning()
                    errorMessageInput = iGetParpoolLinkForError();
                    err = MException(message('parallel:lang:parfor:SessionShutDown', errorMessageInput));
                    err = addCause(err, E);
                    throw(err);
                else
                    rethrow(E);
                end
            end
            % Re-submit pending intervals. Incomplete intervals are non-empty entries in
            % the IncompleteIntervalsCell array.
            gotIncomplete = ~cellfun(@isempty, obj.IncompleteIntervalsCell);
            tags = find(gotIncomplete);
            dataCell = obj.IncompleteIntervalsCell(gotIncomplete);
            
            % Reset IncompleteIntervalsCell
            obj.IncompleteIntervalsCell = cell(size(obj.IncompleteIntervalsCell));
            
            dctSchedulerMessage(2, 'Resubmitting intervals with tags: %s\n', ...
                join(string(tags)));
            for idx = 1:numel(tags)
                OK = obj.addInterval(tags(idx), dataCell{idx}{:});

                % If we've been interrupted, and the pool is down, there is
                % no point in continuing
                if ~OK && ~obj.isSessionValidAndRunning()
                    errorMessageInput = iGetParpoolLinkForError();
                    error(message('parallel:lang:parfor:SessionShutDown', errorMessageInput));
                end
                % No other reasons why this should happen
                assert(OK, 'Unexpected failure to add interval %d.', idx);
            end
            if obj.AllIntervalsAdded
                obj.ParforController.allIntervalsAdded();
            end
        end
        
        function OK = addInterval(obj, tag, varargin)
            try
                obj.IncompleteIntervalsCell{tag} = varargin;
                hasBeenInterrupted = ~obj.ParforController.addInterval(tag, parallel.internal.pool.optionallySerialize(varargin));
                obj.NumIntervalsInController = 1 + obj.NumIntervalsInController;

                % If we've been interrupted and the session has been shut
                % down, we return false, since there is no way the parfor
                % can succeed without a pool. Otherwise, we always return
                % true, so that parallel_function continues. In the case
                % where the controller has already received an error return
                % from an interval, then getCompleteIntervals will notice
                % that error, and either confine it (in the case of a
                % missing source file), or throw it. If
                % getCompleteIntervals can confine the error, then
                % intervals in IncompleteIntervalsCell will be added to a
                % new controller, so we must always store interval requests
                % there even if the current controller rejects them.
                if hasBeenInterrupted && ~obj.isSessionValidAndRunning()
                    OK = false;
                else
                    OK = true;
                end                
            catch E
                dctSchedulerMessage(0, 'Problem in addInterval:', E);
                OK = false;
            end
        end
        
        function allIntervalsAdded(obj)
            obj.ParforController.allIntervalsAdded();
            obj.AllIntervalsAdded = true;
        end
        
        function [tags, results] = getCompleteIntervals(obj, numIntervals)
            
            tags = nan(numIntervals, 1);
            results = cell(numIntervals, 2);
            for i = 1:numIntervals
                err = [];
                r = parallel.internal.pool.ParforIntervalResult();
                while r.isEmpty()
                    assert(obj.NumIntervalsInController > 0, ...
                        'Internal error in PARFOR - no intervals to retrieve.');
                    r = obj.ParforController.waitForNextCompletedInterval(obj.AwaitIntervalPollPeriod);
                    
                    % In each poll tick of the polling wait, yield to any
                    % events allowed to interrupt the parallel language.
                    parallel.internal.pool.yield();
                    if r.isEmpty()
                        % Only test to see if the session is failing if we didn't get a
                        % results from the queue
                        if ~obj.isSessionValidAndRunning()
                            errorMessageInput = iGetParpoolLinkForError();
                            error(message('parallel:lang:parfor:SessionShutDown', errorMessageInput));
                        end
                    else
                        obj.NumIntervalsInController = obj.NumIntervalsInController - 1;
                        if r.hasError()
                            % Maybe try again
                            [r, err] = obj.handleIntervalErrorResult(r);
                        end
                    end
                end
                % Check to see if the interval result has an error
                if r.hasError()
                    throw(err);
                else
                    tags(i) = r.getTag();
                    data = parallel.internal.pool.optionallyDeserialize(r.getResult());
                    assert(numel(data) == 2, ...
                        'Unexpectedly received the incorrect number of outputs.');
                    results(i,:) = data;
                    obj.IncompleteIntervalsCell{tags(i)} = [];
                end
            end
        end
        
        function complete(obj, errorWasThrown)
            dctSchedulerMessage(4, 'ParforEngine.complete(%d) called.', errorWasThrown);
            
            if ~isempty(obj.ParforController)
                lastCall = true;
                obj.ParforController.flushIO(lastCall);
                if errorWasThrown
                    obj.ParforController.interrupt();
                end
            end
            
            obj.NormalCompletion = ~errorWasThrown;
            
            % Note that the yield() call in the loop can cause the
            % ParforController to become empty.
            while ~isempty(obj.ParforController) && ...
                    ~obj.ParforController.awaitCompleted(obj.AwaitCompletedPollPeriod) ...
                    &&  obj.isSessionValidAndRunning()
                parallel.internal.pool.yield();
            end

            % Yield one final time to ensure all events directly generated by
            % code in the PARFOR loop are executed before we give control back
            % to the user.
            parallel.internal.pool.yield();
        end
        
        
        
        function [r, err] = handleIntervalErrorResult(obj, r)
            % handleIntervalErrorResult - called while looping in getCompleteIntervals.
            % Returns empty in r to indicate that this error has been confined.
            
            dctSchedulerMessage(2, 'Got interval error for tag: %d', r.getTag());
            
            rebuildAndRetry = false;
            
            [err, workerAborted] = iIntervalErrorDispatch(r);
            if workerAborted
                dctSchedulerMessage(1, 'Worker aborted during parfor, will attempt to retry.');
                warning(message('MATLAB:remoteparfor:ParforWorkerAborted'));
                rebuildAndRetry = true;
            else
                [err, possibleSourceFiles] = ...
                    parallel.internal.parfor.maybeTransformMissingSourceException(...
                    err, obj.ParforF);
                if ~isempty(possibleSourceFiles) && ~obj.AddedFilesToPool
                    % We've added stuff to the pool, build a new controller and send the intervals.
                    try
                        rebuildAndRetry = obj.addMissingFilesToPool(possibleSourceFiles);
                    catch E
                        % Attaching files failed, log and continue - rebuildAndRetry
                        % will remain false.
                        dctSchedulerMessage(1, 'Failed to addMissingFilesToPool.', E);
                    end
                end
            end
            if rebuildAndRetry
                dctSchedulerMessage(1, 'Rebuilding parfor controller for retry.');
                obj.rebuildParforController();
                r = parallel.internal.pool.ParforIntervalResult();
                err = obj.EMPTY_EXCEPTION;
            end
        end
        
        function [init, remain] = getDispatchSizes(~, numIntervals, W)
            init = iClamp(parallel.internal.parfor.ParforEngine.initialDispatchFactor() * W, ...
                1, numIntervals);
            remain = 2 * W;
        end
        
        function W = getNumWorkers(obj)
            W = obj.NumWorkers;
        end
        
        function filesAttached = addMissingFilesToPool(obj, possibleSourceFiles)
            % Before we even try, let's be sure not to do this again.
            obj.AddedFilesToPool = true;
            filesAttached = parallel.internal.pool.attachDependentFilesToPool(...
                obj.Pool, possibleSourceFiles);
        end
        
        function delete(obj)
            if obj.NormalCompletion || isempty(obj.ParforController)
                % No need to interrupt nor drain output
                return;
            end
            
            % Send a Ctrl+C to remote end.
            obj.ParforController.interrupt();
            
            while ~obj.ParforController.awaitCompleted(obj.AwaitCompletedPollPeriod) ...
                    && obj.isSessionValidAndRunning()
                parallel.internal.pool.yield();
            end
        end
    end

    methods (Access = private)
        function tf = isSessionValidAndRunning(obj)
            tf = isvalid(obj.Session) && obj.Session.isSessionRunning();
        end
    end
    
    methods(Static)
        function oldVal = maxBacklog(newVal)
            % 'Backlog' refers to how many intervals the ParforController will send to the
            % workers without requiring a result.
            %
            % See the discussion in 'numIntervalsFactor' for how this factor
            % interacts with that value.
            persistent MAX_BACKLOG
            if isempty(MAX_BACKLOG)
                MAX_BACKLOG = 2;
            end
            oldVal = MAX_BACKLOG;
            if nargin > 0
                MAX_BACKLOG = newVal;
            end
        end
        
        function oldVal = initialDispatchFactor(newVal)
            % parallel_function.m uses getInitialDispatchSize() to choose how many intervals
            % to hand off in the first round of dispatch. That method scales the
            % number of workers by this factor to choose the initial dispatch
            % size. The default is to allow parallel_function to dispatch all
            % intervals up-front. This allows ParforEngine to receive all
            % the intervals as soon as possible and then send them to the workers.
            %
            % This factor is provided to allow an approximation of the old behavior
            % to be achieved; however, it is not anticipated that there is any
            % appreciable benefit to changing this value.
            persistent FACTOR
            if isempty(FACTOR)
                FACTOR = Inf;
            end
            oldVal = FACTOR;
            if nargin > 0
                FACTOR = newVal;
            end
        end
    end
end

function val = iClamp(inVal, minBound, maxBound)
val = min(max(inVal, minBound), maxBound);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Return an appropriate error from a failing interval result.
function [err, workerAborted] = iIntervalErrorDispatch(r)
assert(r.hasError())

workerAborted = r.didWorkerAbort();
if workerAborted
    err = MException(message('parallel:lang:pool:WorkerAborted'));
elseif r.hasRuntimeError()
    origErr = parallel.internal.pool.handleRemoteException(r.getRuntimeError());
    err = iCreateRemoteException(origErr);
else
    % The result is the MATLAB error
    origErr = parallel.internal.pool.optionallyDeserialize(r.getResult());
    err = iCreateRemoteException(origErr);
end
end

function err = iCreateRemoteException(origErr)
    clientStackToIgnore = 'parallel_function';
    % Create a new exception that stitches together the client and worker stack.
    err = ParallelException.hBuildRemoteParallelException(...
        origErr, clientStackToIgnore);
end

function iErrorCheckPool(pool)
if isempty(pool) || ~pool.Connected
    errorMessageInput = iGetParpoolLinkForError();
    error(message('parallel:lang:parfor:NoSession', errorMessageInput));
end
end

function errorMessageInput = iGetParpoolLinkForError()
if feature('hotlinks')
    errorMessageInput = '<a href="matlab: helpPopup parpool">parpool</a>';
else
    errorMessageInput = 'parpool';
end
end
