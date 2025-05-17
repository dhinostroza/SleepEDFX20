%CppBackedSession Implementation of parallel.internal.pool.ISession
%

% Copyright 2020-2024 The MathWorks, Inc.

classdef CppBackedSession < parallel.internal.pool.ISession

    properties (Constant)
        % Maximum duration to wait for client/worker to bind to port
        % This time should not need to scale with pool size.
        EndpointBindTimeout = seconds(30);

        % Minimum timeout to wait for client/worker to connect to remote 
        % port. Will be increased with more workers, see next timeout.
        BaseEndpointConnectTimeout = minutes(2);

        % Amount to increase above connect timeout per 100 workers in pool.
        % This accounts for the extra load on the client which will delay
        % the connect handshake
        EndpointConnectTimeoutScaling = minutes(1) % 1 min per 100 workers;

        % The following timeouts are used by the workers during pool startup.
        % The timeout for the client is controlled independently with
        % pctconfig(). The timeouts are all Inf, since we let the client
        % decide how long to wait, and when to time out.

        % Minimum timeout for a worker to wait for the client/other worker
        % services to appear. Will increase with more workers, see next
        % timeout.
        BaseWorkerWaitForPeerTimeout = seconds(Inf);

        % Amount to increase above connect timeout per 100 workers in pool.
        % This accounts for the extra load on the client which will delay
        % service directory exchange
        WorkerWaitForPeerTimeoutScaling = seconds(Inf);

        % A list of function names that we wish to mask from warning stacks.
        % These correspond to internal details of parallel language constructs.
        FunctionsToMaskFromStack = {'parallel_function', ...
            'parallel.internal.parfor.cppRemoteParallelFunction', ...
            'spmdlang.remoteBlockExecutionPlain', ...
            'spmdlang.remoteBlockExecutionSerDeser', ...
            'parallel.internal.queue.cppEvaluateRequest' };
    end

    properties (Constant)
        % The minimum number of workers we may start a parfor block on
        MinWorkersToStartParfor = 1;

        % The timeout to wait for the above MinWorkersToStartParfor to
        % become available when starting a parfor block before we warn
        ParforReserveTimeout = seconds(60);
    end

    properties (Constant, GetAccess = private)
        % Depending on topology, we use the ValueStore to exchange
        % endpoints, after successfully finding a port to bind to, using
        % keys with these prefixes.
        % Use Internal key names to avoid users ever seeing them
        % For the client to connect to all the workers
        ConnectEndpointsForClientKeyPrefix = parallel.internal.keyvalue.Store.hCreateInternalKey("connectEndpointsForClientFromWorker") % each worker will add e.g. _2
    end
    
    properties (SetAccess = immutable)
        % The UUID associated with this session
        UUID (1,1) string

        % Whether this is an interactive parpool or a batch parpool
        IsInteractive;

        % The C++ session object
        SpfSession

        % How the pool is connected
        PoolTopology
    end

    properties (SetAccess = private)
        SessionInfo
        CompositeKeysToClearMap = []
        ConnectAcceptController = []
        MPIController = [];
        PathAssistant
        FileDependenciesAssistant = []
        LicensingListener
        RemoteDataQueueAccess = [];

        % The endpoint(s) we use for pool communication. Depending on
        % topology and role, we might have both bind and connect endpoints.
        % Only used for testing.
        BindEndpoint
        ConnectEndpoints

    end

    methods (Access = private)
        function obj = CppBackedSession(uuid, isInteractive, session, poolTopology)
            obj.UUID = uuid;
            obj.IsInteractive = isInteractive;
            obj.SpfSession = session;
            obj.PoolTopology = poolTopology;
        end
    end

    methods (Access = {?parallel.internal.pool.AbstractClusterPool, ...
                       ?parallel.internal.pool.ISession})
        function uuid = getSessionUUID(obj)
            uuid = obj.UUID;
        end

        function tf = canInitializeSpmd(obj)
            tf = obj.SpfSession.canInitializeSpmd();
        end

        function tf = initializeSpmd(obj)
            tf = obj.SpfSession.initializeSpmd();
        end

        function fq = getFevalQueue(obj, pool)
            parfevalController = parallel.internal.queue.ParfevalController(obj.SpfSession, pool);
            fq = parfevalController.getQueueImpl();
            fq.setParfevalController(parfevalController);
        end
        function t = getDefaultWorkerAcquisitionTimeoutMillis(obj)
            t = obj.SpfSession.getDefaultWorkerAcquisitionTimeoutMillis();
        end
    end

    methods
        function didBecomeIdle = waitUntilPoolIdle(obj, timeoutMillis)
            didBecomeIdle = obj.SpfSession.waitUntilPoolIdle(timeoutMillis);
        end

        function ensureAllDataQueueMessagesDelivered(obj)
            % Wait indefinitely for all in-flight DataQueue messages to
            % reach their destination.
            obj.SpfSession.ensureAllDataQueueMessagesDelivered();
        end
    end

    methods (Access = {?parallel.internal.profiler.PoolHelper})
        function setPoolProfilerDataRetrieved(obj, dataRetrieved)
            obj.SpfSession.setPoolProfilerDataRetrieved(dataRetrieved);
        end
    end

    methods (Access = {?parallel.pool.TicBytesResult, ...
                       ?parallel.internal.pool.ISession})
        function info = getCurrentBytesTransferredToInstances(obj)
            info = containers.Map('KeyType', 'double', 'ValueType', 'any');
            m = obj.SpfSession.getTransportByteCounts();
            instances = m.RemoteProcesses;
            for ii = 1:numel(instances)
                index = instances(ii);
                % The client might be included in the list of instances and will have a
                % spmdIndex of -1. Ignore this when building the list of data.
                if index < 1
                    continue
                end
                % Order the information as BytesSentToWorkers followed by
                % BytesReceivedFromWorkers.
                info(index) = [m.getBytesSentTo(index), m.getBytesReceivedFrom(index)];
            end
        end
    end

    methods (Access = {?parallel.internal.pool.AbstractClient, ...
                       ?parallel.internal.pool.ISession})
        function isShutdownSuccessful = destroyClientSession(obj)
            % This will wait for shutdown to complete
            % Return true if pool workers were successfully shutdown and we
            % anticipate the pool job will finish shortly.
            isShutdownSuccessful = obj.shutdown();
        end

        function startSendPathAndClearNotificationToLabs(obj, syncPackageRegistry)
            % TODO g2702048 remove syncPackageRegistry parameter when supported everywhere
            obj.PathAssistant = parallel.internal.pool.PathAssistant(obj.SpfSession, syncPackageRegistry);
        end

        function waitForConnections(obj, spmdEnabled, checkFcn, job)
            import parallel.internal.spf.ConnectionTopology

            % Pool clients end up here with a dummy sessionInfo, we need a real one.
            if isempty(obj.SessionInfo) || strcmp(obj.SessionInfo.State, 'CLOSED')
                obj.SessionInfo = iCreateStartingSessionInfo();
            end
            obj.SpfSession.setSessionInfo(obj.SessionInfo.CppSessionInfo);
            parallel.internal.pool.bootstrapClientServices(obj.SpfSession, obj.IsInteractive);

            printer = parallel.internal.pool.ParpoolProgressPrinter(obj.SessionInfo.CppSessionInfo, job.ID);
            printingCheckFcn = @() iPrintingCheckFcn(printer, checkFcn);
            % Don't start waiting for connections before the job is running
            while ~wait(job, 'running', 0.1)
                printingCheckFcn();
            end
            parallel.internal.pool.JobStateChecker.throwIfBadParallelJobStatus(job);
            obj.SessionInfo.CppSessionInfo.setJobRunning();

            clientNeedsToConnect = obj.PoolTopology == ConnectionTopology.CLIENT_TO_WORKERS || ...
                obj.PoolTopology == ConnectionTopology.CLIENT_TO_WORKERS_AND_WORKERS_TO_LEAD_WORKER;

            poolStartTimeout = pctconfig().poolstarttimeout;
            error = MException(message("parallel:lang:pool:TimeoutSettingUpSessionOnClient", seconds(poolStartTimeout)));

            if clientNeedsToConnect
                dctSchedulerMessage(4, "Client starting to connect to workers");
                allEndpoints = iConnectToEndpoints(job.ValueStore, ...
                    obj.ConnectEndpointsForClientKeyPrefix, ...
                    printingCheckFcn, obj.SpfSession, ...
                    poolStartTimeout, ...
                    error, obj.SessionInfo.CppSessionInfo);
                obj.ConnectEndpoints = allEndpoints;
            end
            iWaitForAllWorkerServices(obj.SpfSession, spmdEnabled, printingCheckFcn, poolStartTimeout);
            
            % Setup dataqueue access to remote data
            obj.RemoteDataQueueAccess = parallel.internal.dataqueue.SpfDataQueueExchangeAccess(obj.SpfSession);
        end
    end

    methods
        function evalOnClient(obj, cmd)
            parallel.internal.pool.evalOnClient(obj.SpfSession, cmd);
        end

        function fda = getFileDependenciesAssistant(obj)
            if isempty(obj.FileDependenciesAssistant)
                obj.FileDependenciesAssistant = ...
                    parallel.internal.pool.FileDependenciesAssistant(obj.SpfSession);
            end
            fda = obj.FileDependenciesAssistant;
        end

        function n = getPoolSize(obj)
            n = double(obj.SessionInfo.SessionSize);
        end

        function tf = isSessionRunning(obj)
            tf = obj.SpfSession.isSessionRunning();
        end

        function setRestartOnClusterChange(obj, tf)
            obj.SessionInfo.setRestartOnClusterChange(tf);
        end

        function setShutdownIdleTimeout(obj, timeInSeconds)
            obj.SpfSession.setIdleTimeout(timeInSeconds);
        end

        function info = getClientSessionInfo(obj)
            info = obj.SessionInfo;
        end

        function preRemoteEvaluationCheck(obj)
            obj.SpfSession.preRemoteEvaluationCheck();
        end
                
        function releaseCurrentParforController(obj)
            assert(false, 'Unimplemented.');
        end

        function controller = createParforController(obj, maxBacklog, maxNumWorkers)
            controller = parallel.internal.pool.ParforController(...
                    obj.SpfSession, maxBacklog, ...
                    obj.MinWorkersToStartParfor, maxNumWorkers, ...
                    milliseconds(obj.ParforReserveTimeout));
            obj.ensureAllDataQueueMessagesDelivered();
        end

        function controller = createSpmdController(obj, workerIndices)
            controller = parallel.internal.pool.RemoteSpmdControllerImpl.create(...
                obj.SpfSession, workerIndices, obj.getCompositeKeysToClearMap());
            obj.ensureAllDataQueueMessagesDelivered();
        end
        
        function map = getCompositeKeysToClearMap(obj)
            if isempty(obj.CompositeKeysToClearMap)
                obj.CompositeKeysToClearMap = parallel.internal.pool.ProcessKeyMap();
            end
            map = obj.CompositeKeysToClearMap;
        end

        function assistant = getCompositeAssistant(obj, workerIndices)
            assistant = parallel.internal.pool.RemoteCompositeAssistantImpl.createForSession(...
                obj.SpfSession, workerIndices, obj.getCompositeKeysToClearMap());
        end

        function runConnectAccept(obj, pool, startedFcn)
            if isempty(obj.ConnectAcceptController)
                obj.ConnectAcceptController = parallel.internal.pool.ConnectAcceptController(obj.SpfSession);
            end
            mpiOption = parallel.internal.pool.connectAccept.computeMpiOption(pool);
            spmdInvalidGuard = parallel.internal.general.DisarmableOncleanup(@() iMaybeInvalidateSPMD(startedFcn, obj.ConnectAcceptController));
            warningTimeout = ceil(milliseconds(pool.SpmdInitializationWarningTime));
            obj.ConnectAcceptController.runConnectAccept(mpiOption, warningTimeout);
            spmdInvalidGuard.disarm();
        end

        function controller = getMPIController(obj)
            if isempty(obj.MPIController)
                % Check which mpi communicator covers all worker
                % processes in the pool. For an interactive pool this will
                % be the "world", for a batch pool this will be the split
                % communicator including all processes except the lead
                % worker which is the pool client. The code which may set
                % this is in PoolClient.
                [worldComm, workerIdxOfFirst] = spmdlang.commForWorld('get');
                if ischar(worldComm) % string 'world' maps to key 0
                    numWorkers = obj.getPoolSize();
                    worldComm = zeros(1, numWorkers, 'uint64');
                end
                obj.MPIController = parallel.internal.pool.ProcessesMPIController.createForSession(worldComm, workerIdxOfFirst, obj.SpfSession);
            end
            controller = obj.MPIController;
        end

        function factory = createRemoteQueueFactory(obj, queueUuid)
        % When building the remote queue factory, we embed in the function
        % handle that we provide the necessary internal information needed to
        % build the RemoteSpfQueue.
            sessionUuid = obj.UUID;
            receiverProcessUuid = obj.SpfSession.ProcessUuid;
            buildFcn = @() parallel.internal.pool.CppBackedSession.buildRemoteSpfQueue(...
                sessionUuid, queueUuid, receiverProcessUuid);
            factory = parallel.internal.dataqueue.RemoteSpfQueueFactory(obj, buildFcn);
        end

        function assistant = getConstantAssistant(obj)
            assistant = parallel.internal.constant.ConstantAssistant.createForSpfSession(obj.SpfSession);
        end

        function isShutdownSuccessful = shutdown(obj)
            isShutdownSuccessful = obj.SpfSession.shutdown();
        end
    end

    methods (Static, Access = private)
        function s = findSessionForUuid(sessionUuid)
        % Find instance of this class on client or worker if available. This
        % is used by DataQueue which needs to send a reference to a specific
        % pool session.
            s = [];
            sessionObject = parallel.internal.pool.workerSession();
            if isempty(sessionObject)
                pool = parallel.internal.pool.PoolArrayManager.getCurrentWithCleanup();
                if ~isempty(pool) && ~isa(pool, "parallel.ThreadPool")
                    sessionObject = pool.hGetClient();
                end
            end
            if isempty(sessionObject)
                return
            end
            possibleSession = sessionObject.Session;
            if possibleSession.isSessionRunning() && isequal(possibleSession.UUID, sessionUuid)
                s = possibleSession;
            end
        end

        function t = getEndpointConnectTimeoutMilliseconds(numWorkers)
            % How long should we wait to connect to the client/other worker?
            % Since each connect involves a handshake, we scale this time
            % by the number of workers.
            t = parallel.internal.pool.CppBackedSession.BaseEndpointConnectTimeout + ...
                (parallel.internal.pool.CppBackedSession.EndpointConnectTimeoutScaling * numWorkers/100);
            t = ceil(milliseconds(t));
        end

        function t = getWorkerWaitForPeerTimeout(numWorkers)
            % How long should we wait for client/other worker services to appear?
            % Since this involves a directory handshake, we scale this time
            % by the number of workers.
            t = parallel.internal.pool.CppBackedSession.BaseWorkerWaitForPeerTimeout + ...
                ceil(parallel.internal.pool.CppBackedSession.WorkerWaitForPeerTimeoutScaling * numWorkers/100);
        end
    end

    methods (Static)
        function rq = buildRemoteSpfQueue(sessionUuid, queueUuid, receiverProcessUuid)
            s = parallel.internal.pool.CppBackedSession.findSessionForUuid(sessionUuid);
            if isempty(s)
                error(message('MATLAB:parallel:dataqueue:InvalidProcess'));
            end
            rq = parallel.internal.dataqueue.RemoteSpfQueue(...
                queueUuid, receiverProcessUuid, s.SpfSession);
        end
        
        function [obj, taskArgs] = buildInteractiveClient(cluster, sessionInfo)
            import parallel.internal.spf.ConnectionTopology

            uuid = sessionInfo.SessionId;
            dctSchedulerMessage(2, 'Setting up for interactive client session %s', uuid);
            endpointFactory = cluster.hCreatePoolEndpointFactory();

            isInteractiveClient = true;
            clientConnects = endpointFactory.PoolTopology == ConnectionTopology.CLIENT_TO_WORKERS || ...
                endpointFactory.PoolTopology == ConnectionTopology.CLIENT_TO_WORKERS_AND_WORKERS_TO_LEAD_WORKER;
            if clientConnects
                % Create client session which will later connect to worker
                % endpoints
                clientSession = parallel.internal.pool.SpfClientSession(parallel.internal.spf.ClientEndpointWrapper.empty(), uuid, ...
                    parallel.internal.pool.CppBackedSession.getEndpointConnectTimeoutMilliseconds(1));

                obj = parallel.internal.pool.CppBackedSession(uuid, isInteractiveClient, clientSession, endpointFactory.PoolTopology);

                taskArgs = {@(workerIdx, numWorkers, ~) ...
                            parallel.internal.pool.CppBackedSession.buildWorker( ...
                    [], workerIdx, numWorkers, uuid, endpointFactory, isInteractiveClient), ...
                            @() iYieldLoop(uuid)};
            else % client binds
                [bindEndpoint, connectEndpoint] = endpointFactory.createEndpointsForClient();
                
                clientSession =  parallel.internal.pool.SpfClientSession(bindEndpoint, ...
                        uuid, milliseconds(parallel.internal.pool.CppBackedSession.EndpointBindTimeout));
            
                dctSchedulerMessage(4, "Interactive client bound to URL: %s and port %d", bindEndpoint.URL, bindEndpoint.Port);

                % Update connectEndpoint with the port selected
                connectEndpoint.Port = bindEndpoint.Port;

                obj = parallel.internal.pool.CppBackedSession(uuid, isInteractiveClient, clientSession, endpointFactory.PoolTopology);
                taskArgs = {@(workerIdx, numWorkers, ~) ...
                            parallel.internal.pool.CppBackedSession.buildWorker( ...
                    connectEndpoint, workerIdx, numWorkers, uuid, endpointFactory, isInteractiveClient), ...
                            @() iYieldLoop(uuid)};

                obj.BindEndpoint = bindEndpoint;
            end
            obj.SessionInfo = sessionInfo;
        end

        function obj = buildWorker(connectToClientEndpoint, workerIdx, numWorkers, sessionUuid, endpointFactory, isInteractiveSession)
            import parallel.internal.spf.ConnectionTopology

            % Based on topology and worker index we might need to do one or
            % more of the following actions:
            % Bind to a port for other workers/the client to connect to and
            % put the selected port in the value store.
            % ConnectToClient using the endpoint provided as an argument to
            % this function.
            % ConnectToLeadWorker after finding the selected port in the
            % value store.
            % We also set numExpectedPeers, i.e. the number of other
            % clients/workers we will be directly connected to.

            % Special cases for 1 worker pools
            topology = endpointFactory.PoolTopology;

            if numWorkers == 1
                topology = topology.getOneWorkerPoolEquivalent();
            end

            [workersConnectToClient, workersConnectToLeadWorker] = iGetPoolConnectionActions(topology);
            dctSchedulerMessage(4,"Building pool worker with topology: %s", char(endpointFactory.PoolTopology));

            [bindEndpoint, connectEndpointForClient, connectEndpointForOtherWorkers] = endpointFactory.createEndpointsForWorker();
            bindEndpoint = iSetPortSelectionInformation(bindEndpoint);

            % Always bind on workers to allow dynamic point-to-point
            % connections between workers
            dctSchedulerMessage(4, "About to bind to port");
            workerSession =  parallel.internal.pool.SpfWorkerSession.createWithServerEndpoint(bindEndpoint, ...
                sessionUuid, workerIdx, milliseconds(parallel.internal.pool.CppBackedSession.EndpointBindTimeout));
            obj = parallel.internal.pool.CppBackedSession(sessionUuid, isInteractiveSession, workerSession, topology);
            obj.BindEndpoint = bindEndpoint;

            % Set the selected port on the connect endpoints
            connectEndpointForClient.Port = bindEndpoint.Port;
            connectEndpointForOtherWorkers.Port = bindEndpoint.Port;

            % Check whether we should proxy client-worker connections
            % through SPF proxy process. Never proxy client-worker pool
            % traffic if:
            % 1. This is a batch pool where the client must be a worker
            %    in the cluster
            % 2. The workers connect to the client server socket
            % 3. The interactive pool client is in fact a worker in the
            %    cluster
            shouldCheckIfProxyUsed = isInteractiveSession && ...
                ~workersConnectToClient && ...
                ~endpointFactory.ClientInCluster;
            if shouldCheckIfProxyUsed
                [useProxy, connectToProxyEndpoint, proxyEndpointForClient] = endpointFactory.createProxyEndpointIfProxyingEnabled();
            else
                useProxy = false;
                connectToProxyEndpoint = [];
            end

            connectEndpoints = [];

            % Connect worker to proxy if needed
            if useProxy
                connectEndpoints = connectToProxyEndpoint;
                dctSchedulerMessage(4, "Connecting to proxy at %s", connectToProxyEndpoint.URL);
                obj.SpfSession.addClientEndpointsAndConnect(connectToProxyEndpoint, ...
                    parallel.internal.pool.CppBackedSession.getEndpointConnectTimeoutMilliseconds(1));
                connectEndpointForClient = proxyEndpointForClient;
            end

            if workersConnectToClient
                % Worker connects to client
                connectToClientEndpoint = iProcessClientHostOverride(connectToClientEndpoint);
                connectEndpoints = [connectEndpoints connectToClientEndpoint];
                dctSchedulerMessage(4, "Connecting to client at %s", connectToClientEndpoint.URL);
                obj.SpfSession.addClientEndpointsAndConnect(connectToClientEndpoint, ...
                    parallel.internal.pool.CppBackedSession.getEndpointConnectTimeoutMilliseconds(1));
            else
                % Publish connect endpoint for client to use to connect to
                % workers
                keyForClient = sprintf("%s_%d", parallel.internal.pool.CppBackedSession.ConnectEndpointsForClientKeyPrefix, workerIdx);
                valueStore = getCurrentValueStore();
                valueStore(keyForClient) = {connectEndpointForClient, numWorkers}; %#ok<NASGU> handle class
                dctSchedulerMessage(4, "Placed endpoint for pool client in ValueStore for worker %d ", workerIdx);
            end

            dctSchedulerMessage(4, "About to start waiting for client formation")
            iWaitForServicesOrError(@(x) obj.SpfSession.waitForClientFormation(x), numWorkers);
            dctSchedulerMessage(4, "Successfully waited for client formation")

            % Register worker with client
            registrationResponse = obj.SpfSession.registerWorker(connectEndpointForOtherWorkers);

            % Connect to lead worker if required
            isLeadWorker = workersConnectToLeadWorker && ~registrationResponse.hasLeadWorkerConnectEndpoint();
            if workersConnectToLeadWorker
                if isLeadWorker
                    % We are the first worker to register and hence the
                    % lead worker
                    dctSchedulerMessage(4, "Allocated as lead worker");
                else
                    leadWorkerConnectEndpoint = registrationResponse.getLeadWorkerConnectEndpoint();
                    connectEndpoints = [connectEndpoints leadWorkerConnectEndpoint];
                    dctSchedulerMessage(4, "Connecting to lead worker at %s", leadWorkerConnectEndpoint.URL);
                    obj.SpfSession.addClientEndpointsAndConnect(leadWorkerConnectEndpoint, ...
                        parallel.internal.pool.CppBackedSession.getEndpointConnectTimeoutMilliseconds(1));
                end
            end

            obj.ConnectEndpoints = connectEndpoints; 

            % If this is a worker session and it is part of an interactive pool, start an instance of the LicenseNotificationListener.
            if isInteractiveSession
                obj.LicensingListener = parallel.internal.pool.LicenseNotificationListener(obj.SpfSession);
            end

            if workersConnectToLeadWorker && ~isLeadWorker
                % Ensure we are connected to the lead worker before
                % publishing our services. This ensures all workers the
                % client knows about are also reachable via the lead
                % worker.
                dctSchedulerMessage(4, "About to start to wait for lead worker");
                iWaitForServicesOrError(@(x) obj.SpfSession.waitForLeadWorker(x), numWorkers);
            end
            dctSchedulerMessage(4, "Starting worker services, isleadWorker: %d", isLeadWorker);
            t = getCurrentTask();
            workerUserDescription = iGetWorkerUserDescription(workerIdx, t.ID, endpointFactory.getWorkerConnectHostname());
            parallel.internal.pool.bootstrapWorkerServices(obj.SpfSession, isLeadWorker, numWorkers, getAttachedFilesFolder(), workerUserDescription);
            
            % Setup dataqueue access to remote data
            obj.RemoteDataQueueAccess = parallel.internal.dataqueue.SpfDataQueueExchangeAccess(obj.SpfSession);

            % Set up the persistent state for workerYield:
            parallel.internal.pool.CppBackedSession.workerYield(obj.SpfSession);
            % Tell parallel.internal.pool.poolWorkerFcn to call this function
            % for synchronous pool evaluation. This environment variable is
            % also read by dctEvaluateTask to discover that the pool execution
            % is synchronous.
            setenv('MDCS_SYNC_TASK_FCN', ...
                   'parallel.internal.pool.CppBackedSession.workerYield');
        end

        function [obj, workerBuild] = buildBatchClient()
            uuid = parallel.internal.pool.SpfClientSession.allocateSessionId();
            dctSchedulerMessage(2, 'Setting up for batch client session %s', uuid);

            cluster = getCurrentCluster();
            endpointFactory = cluster.hCreatePoolEndpointFactory();

            % For batch pools we ignore the topology the endpointFactory
            % has determined, and always use the direct topology.
            endpointFactory.enforceWorkersToClientTopology();
            endpointFactory.ClientInCluster = true; % The client and workers are always in the same cluster in a batch pool
            
            % The serverEndpoint is created with a port range, but the port
            % is unset until we find a specific port to bind to when we
            % create the client session
            [bindEndpoint, connectEndpoint] = endpointFactory.createEndpointsForClient();
            clientSession =  parallel.internal.pool.SpfClientSession(bindEndpoint, ...
                                                                     uuid, milliseconds(parallel.internal.pool.CppBackedSession.EndpointBindTimeout));
            isInteractiveClient = false;
            obj = parallel.internal.pool.CppBackedSession(uuid, isInteractiveClient, clientSession, endpointFactory.PoolTopology);

            dctSchedulerMessage(4, "Batch client bound to URL: %s and port %d", bindEndpoint.URL, bindEndpoint.Port);
            connectEndpoint.Port = bindEndpoint.Port;
            workerBuild = @(workerIdx, numWorkers, ~) ...
                parallel.internal.pool.CppBackedSession.buildWorker( ...
                connectEndpoint, workerIdx, numWorkers, uuid, endpointFactory, isInteractiveClient);
            
            obj.BindEndpoint = bindEndpoint;
        end

        function workerYield(sessionObj)
            persistent SESSION_OBJ
            if nargin == 1
                dctSchedulerMessage(2, 'Setting up for pool session %s', sessionObj.UUID);
                SESSION_OBJ = sessionObj;
            else
                % Swap out SESSION_OBJ to ensure it doesn't outlive this function.
                sessionObj = SESSION_OBJ;
                SESSION_OBJ = [];
                iYieldLoop(sessionObj);
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sessionInfo = iCreateStartingSessionInfo()
    % We don't have cluster or profile info here, but it is not needed.
     try
         job = getCurrentJob();
         clusType = job.Parent.Type;
         clusProf = job.Parent.Profile;
     catch E
         dctSchedulerMessage(2, 'Failed to get info from job.', E);
         clusType = '';
         clusProf = '';
     end
     sessionInfo = parallel.internal.pool.CppBackedSessionInfo();
     sessionInfo.notifyAboutToStart(clusType, clusProf);
     sessionInfo.notifyStarting(clusType, clusProf, intmax('int64'), false);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function iYieldLoop(workerSession)
    % Ensure warning stacks issued by workers don't include this
    % frame, or any of our callers.
    cleanup = iMaskStackFrames(); %#ok<NASGU> onCleanup object to restore state
    uuid = workerSession.UUID;
    dctSchedulerMessage(2, 'Worker is ready to execute commands for pool session %s', uuid);
    while workerSession.isSessionRunning()
        % Use simply 'pause' to process all IQM events in chunks.
        pause(0.1);
    end
    dctSchedulerMessage(2, 'Pool session %s no longer running.', uuid);

    workerSession.shutdown();
    
    % Begin worker shutdown
    parallel.internal.pool.stopInteractiveWorker();
    workerSession.stopWorkerTerminationThread();
    dctFinishInteractiveSession();
end

%#ok<*INUSD> unimplemented methods.
%#ok<*MANU> unimplemented methods.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up maskFoldersFromStack to hide infrastructure frames for warnings.
% Returns onCleanup object to revert.
function cleanup = iMaskStackFrames()
    % Get a list of all the files currently in the execution stack. We want to
    % mask all of these.
    stack = dbstack('-completenames');
    stackFiles = unique({stack.file});

    % Here's a list of additional files we wish to mask.
    additionalFiles = cellfun(@which, parallel.internal.pool.CppBackedSession.FunctionsToMaskFromStack, ...
        UniformOutput = false);

    files = unique([stackFiles, additionalFiles]);
    
    % Ensure we choose only things we're adding here.
    orig = matlab.lang.internal.getMaskedFoldersFromStack();
    toAdd = setdiff(files, orig);

    % Call maskFoldersFromStack, with cleanup.
    cleanup = onCleanup(@() matlab.lang.internal.unmaskFoldersFromStack(toAdd));
    matlab.lang.internal.maskFoldersFromStack(toAdd);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Special case code for the client address - if there happens to be an
% environment variable MDCE_OVERRIDE_CLIENT_HOST, use that in
% preference. This allows the local scheduler to defend against a
% variety of problems, including the client machine having either a
% broken name, or one which clashes with some other machine on the
% network.
function endpoint = iProcessClientHostOverride(endpoint)
    overrideClientHostname = getenv("MDCE_OVERRIDE_CLIENT_HOST");
    if ~isempty(overrideClientHostname)
        dctSchedulerMessage(2, 'Applying MDCE_OVERRIDE_CLIENT_HOST: %s', overrideClientHostname);
        endpoint.Address = overrideClientHostname;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function iWaitForServicesOrError(waitFcn, numWorkers)
    import parallel.internal.timestamps.createDeadline
    import parallel.internal.timestamps.hasDeadlineExpired

    t = tic();
    logGranularity = 1; % Don't log more frequently than this
    maxLogGranularity = 16; % Max allowed log interval
    timeAllowanceSeconds = seconds(parallel.internal.pool.CppBackedSession.getWorkerWaitForPeerTimeout(numWorkers));
    deadline = createDeadline(timeAllowanceSeconds);
    while true
        if waitFcn(1000)
            break
        end
        if hasDeadlineExpired(deadline)
            error(message("parallel:lang:pool:TimeoutSettingUpSession"));
        end
        if toc(t) > logGranularity
            dctSchedulerMessage(6, 'buildWorker waiting for client connection');
            logGranularity = min(maxLogGranularity, 2*logGranularity);
            t = tic();
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function iWaitForAllWorkerServices(session, spmdEnabled, checkFcn, timeout)
    import parallel.internal.timestamps.invalidTime
    import parallel.internal.timestamps.currentTime

    maxRegistrationGap = ...
        ceil(milliseconds(timeout));
    % Wait indefinitely, relying on worker/job timeouts to detect stalls.
    lastTimeOfProgress = currentTime();
    while true
        if session.waitForAllWorkers(1000, spmdEnabled)
            % All workers up
            break;
        end
        mostRecentRegistration = session.timeOfMostRecentWorkerRegistration();
        if mostRecentRegistration ~= invalidTime()
            lastTimeOfProgress = mostRecentRegistration;
        end
        if (currentTime() > (lastTimeOfProgress + maxRegistrationGap))
            % Workers have started registering, but it has been too long since the last
            % registration. Bail out.
            mostRecentDT = datetime(double(lastTimeOfProgress)/1000, 'ConvertFrom', 'posixtime', ...
                                    'TimeZone', 'local');
            dctSchedulerMessage(1, 'Timeout waiting for workers. Most recent worker registration was at: %s', ...
                                string(mostRecentDT));
            error(message("parallel:lang:pool:TimeoutSettingUpSessionOnClient", ...
                seconds(timeout)));
        end
        % checkFcn will abort this function with an error if necessary.
        checkFcn();
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function connectedEndpoints = iConnectToEndpoints(valueStore, keyBase, checkFcn, session, timeout, timeoutError, sessionInfo)
import parallel.internal.timestamps.createDeadline
import parallel.internal.timestamps.hasDeadlineExpired

if nargin < 7
    sessionInfo = [];
end

processedKeys = {};
connectedEndpoints = [];
numExpectedEndpoints = 1;
deadline = createDeadline(seconds(timeout));
dctSchedulerMessage(5, "Starting to wait for endpoints");

while numel(processedKeys) < numExpectedEndpoints
    allKeys = valueStore.hKeys();
    allKeys = allKeys(startsWith(allKeys, keyBase));
    newKeys = setdiff(allKeys, processedKeys);
    
    if ~isempty(newKeys)
        connectEndpoints = valueStore.get(newKeys);
        newNumExpectedEndpoints = connectEndpoints{1}{2};
        if newNumExpectedEndpoints ~= numExpectedEndpoints
            numExpectedEndpoints = newNumExpectedEndpoints;
        end

        connectEndpoints = cellfun(@(x) x{1}, connectEndpoints);
        connectEndpoints = reshape(connectEndpoints, 1, numel(connectEndpoints));
        session.addClientEndpointsAndConnect(connectEndpoints, ...
            parallel.internal.pool.CppBackedSession.getEndpointConnectTimeoutMilliseconds(numel(connectEndpoints)));
        processedKeys = union(processedKeys, newKeys);
        connectedEndpoints = [connectedEndpoints connectEndpoints]; %#ok<AGROW>
        dctSchedulerMessage(5, "Connected to %d of %d endpoints", numel(processedKeys), numExpectedEndpoints);
        if ~isempty(sessionInfo)
            sessionInfo.setNumConnectedWorkers(numel(processedKeys), numExpectedEndpoints);
        end

        % Reset deadline as long as we're making progress
        deadline = createDeadline(seconds(timeout));
    else
        pause(0.1);
        if hasDeadlineExpired(deadline)
            throw(timeoutError);
        end
    end
    valueStore.remove(allKeys);

    % checkFcn will abort this function with an error if necessary.
    checkFcn();
end

dctSchedulerMessage(5, "Connected to all %d endpoints", numExpectedEndpoints);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bindEndpoint = iSetPortSelectionInformation(bindEndpoint)
import parallel.internal.cluster.PortConfig

root = distcomp.getdistcompobjectroot();
portSelectionInformation = root.CurrentRunprop.PortSelectionInformation;

if ~isempty(portSelectionInformation)
    bindEndpoint.IndexOnHost = portSelectionInformation.IndexOnHost;
    bindEndpoint.NumWorkersOnHost = portSelectionInformation.NumWorkersOnHost;
    bindEndpoint.MinPortOffset = portSelectionInformation.MinPortOffset;
    bindEndpoint.PortRangeEnd = double(bindEndpoint.PortRangeStart) + portSelectionInformation.MinPortOffset + ...
        portSelectionInformation.NumWorkersOnHost * PortConfig.getPortsPerWorkerForPool();
else
    dctSchedulerMessage(5, "Did not get PortSelectionInformation, using default values");
    bindEndpoint.IndexOnHost = 0;
    bindEndpoint.NumWorkersOnHost = 1;
    bindEndpoint.MinPortOffset = 0;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [connectToClient, connectToLeadWorker] = iGetPoolConnectionActions(topology)
import parallel.internal.spf.ConnectionTopology

switch topology
    case ConnectionTopology.CLIENT_TO_WORKERS
        [connectToClient, connectToLeadWorker] = deal(false, false);
    case ConnectionTopology.WORKERS_TO_CLIENT
        [connectToClient, connectToLeadWorker] = deal(true, false);
    case ConnectionTopology.CLIENT_TO_WORKERS_AND_WORKERS_TO_LEAD_WORKER
        [connectToClient, connectToLeadWorker] = deal(false, true);
    case ConnectionTopology.WORKERS_TO_CLIENT_AND_WORKERS_TO_LEAD_WORKER
        [connectToClient, connectToLeadWorker] = deal(true, true);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function iPrintingCheckFcn(printer, checkFcn)
checkFcn();
printer.update();
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function iMaybeInvalidateSPMD(startedFcn, controller)
    if ~controller.SPMDPossible
        startedFcn();
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function msg = iGetWorkerUserDescription(index, taskID, host)
    msg = getString(message('parallel:lang:pool:WorkerIndexFull', index, taskID, host));
end
