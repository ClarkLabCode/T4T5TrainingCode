function GenerateXtPlotSinewaveDataset
    %% changeable parameters
    % total number of scenes in the database
    % can't be less than 3 so that there is one natural scene saved for the
    % dev and test sets.
    
    smallestLambda = 20;
    largestLambda = 90;
    
    % how often and how far to sample in space for each plot
    phaseEnd = 360; % degrees
    phaseStep = 5; % degrees
    
    % how often and how far to sample in time
    sampleFreq = 100; % Hz
    totalTime = 1; % s
    
    % number of velocity traces to generate for each point in space
    numTraces = 3040;
    
    % standard devation of gaussian noise (in contrast) to be added to each
    % xt plot
    noiseStd = 0;
    
    % velocity parameters
    % 0.2 is the halflife of the autocorrelation of turning I measured
    % from the data in my 2018 paper
    % 100 degrees/s is teh standard devation of the fly turning from that
    % paper
    halfLife = 0.2; % s
    velStd = 100; % degrees/s

    % fraction of dataset to keep for dev and test sets
    devFrac = 0.05;

    numTime = sampleFreq*totalTime+1;
   
    % make a string to tag the saved file with
    saveStr = [ 'xtPlot_sineWaves' ...
                '_sl' num2str(smallestLambda) ...
                '_ll' num2str(largestLambda) ...
                '_pe' num2str(phaseEnd) ...
                '_ps' num2str(phaseStep) ...
                '_sf' num2str(sampleFreq) ...
                '_tt' num2str(totalTime) ...
                '_nt' num2str(numTraces*2) ...
                '_hl' num2str(halfLife) ...
                '_vs' num2str(velStd) ...
                '_df' num2str(devFrac) ...
                ];
    saveStr(saveStr=='.') = '-';
     
    %% choose phi
    phi = (0:phaseStep:phaseEnd-phaseStep)';
    
    %% get velocities
    tau = halfLife/log(2);
    filterBufferTime = 10*halfLife;
    filterBufferNum = filterBufferTime*sampleFreq;
    
    % assume velocity is normally distributed with a std of 100 in a
    % natural setting. This is based on fly turning rates
    velNoFilter = velStd*randn(filterBufferNum+numTime,numTraces);

    % filter the trace to give it an autocorrelation
    filtT = linspace(0,totalTime+filterBufferTime,filterBufferNum+numTime)';
    % multiply the filter by a corrective constant so that the final trace
    % has the correct std of velocity across traces (but not within a
    % trace). If a trace is particularly short it will look like low
    % variance due to the autocorrelation
    filter = exp(-filtT/tau)*sqrt(1-exp(-2/(tau*sampleFreq)));
    vel = ifft(fft(filter).*fft(velNoFilter));
    vel = vel(1:numTime,:,:);
    
    position = cumsum(vel)/sampleFreq;
    position = position-position(1,:);

    
    
    %% generate xt plots
    xtPlot = zeros(numTime,length(phi),numTraces);
    startPhases = 360*rand(numTraces,1);
    lambdas = smallestLambda + (largestLambda - smallestLambda)*rand(numTraces,1);
    for tt = 1:numTraces
        for pp = 1:length(phi)
            pos = -position(:,tt)+startPhases(tt)+phi(pp);
            xtPlot(:,pp,tt) = sind(pos*360/lambdas(tt));
        end
    end
    
    % add noise
    noise = randn(size(xtPlot))*noiseStd;
    xtPlot = xtPlot + noise;
    
    % create antisymmetrical paired stimuli
    xtPlot = reshape(xtPlot,[numTime,length(phi),1,numTraces]);
    xtPlot = cat(3,xtPlot,fliplr(xtPlot));
    
    vel = reshape(vel,[numTime,1,numTraces]);
    vel = cat(2,vel,-vel);
    
    %% randomize dataset and seperate into train/dev/test

    % fraction of data set to save for dev/test
    devNum = ceil(numTraces*devFrac);
    testNum = devNum;
    trainNum = numTraces-devNum-testNum;

    % get train/dev/train inds
    trainInd = 1:numTraces-2*devNum;

    devInd = numTraces-2*devNum+1:numTraces-devNum;
    testInd = numTraces-devNum+1:numTraces;
    
    % define the test and train x/y pairs
    train_in = xtPlot(:,:,:,trainInd);
    train_in = reshape(train_in,[numTime,length(phi),trainNum*2]);
    
    dev_in = xtPlot(:,:,:,devInd);
    dev_in = reshape(dev_in,[numTime,length(phi),devNum*2]);
    
    test_in = xtPlot(:,:,:,testInd);
    test_in = reshape(test_in,[numTime,length(phi),testNum*2]);

    train_out = vel(:,:,trainInd);
    train_out = reshape(train_out,[numTime,trainNum*2]);
    
    dev_out = vel(:,:,devInd);
    dev_out = reshape(dev_out,[numTime,devNum*2]);
    
    test_out = vel(:,:,testInd);
    test_out = reshape(test_out,[numTime,testNum*2]);

    %% Flip time and space
    train_in = permute(train_in,[2 1 3]);
    dev_in = permute(dev_in,[2 1 3]);
    test_in = permute(test_in,[2 1 3]);
    
    %% save xtPlots
    savePath = fullfile('../data_sets',saveStr);
    save(savePath,'train_in','train_out','dev_in','dev_out','test_in','test_out','sampleFreq','phaseStep','-v7.3');
end
