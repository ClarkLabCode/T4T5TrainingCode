function sinewaves = GenerateSineDataset(velocities,lambda,stimTime,timeOffset,tRes)
	if nargin < 4
        timeOffset = 0;
    end
    if nargin < 5
        tRes = 0.001;
    end
    x = (0:5:355)';
    tts = (0:tRes:stimTime-tRes) + timeOffset;
    [X,T] = meshgrid(x,tts);
    sinewaves = zeros([length(tts),length(x),length(velocities)]);
    for velIdx = 1:length(velocities)
        vel = velocities(velIdx);
        sinewaves(:,:,velIdx) = sind((X-vel*T) * 360/lambda);
    end
end