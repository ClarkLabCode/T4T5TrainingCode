load('intermediate_mat_files/pano.mat')
colormap();

img = projection/max(projection(:));

pixelSizeDeg = 360/size(projection,2);
ommSizePix = 5/pixelSizeDeg;
numReceptors = 10;

receptorStartLocDeg = 180-numReceptors*5/2;
zoomInLocInOrigPix = [receptorStartLocDeg/pixelSizeDeg 90];

cropExt = round(90/pixelSizeDeg):round(270/pixelSizeDeg);

figure(1);
imgCropped = img(:,cropExt);
rectVec = [1 zoomInLocInOrigPix(2)-ommSizePix/2 size(imgCropped,2) ommSizePix];
imgRect = insertShape(imgCropped,'rectangle',rectVec,'lineWidth',3);
imshow(imadjust(imgRect,[0.035 0.4],[],0.5));

figure(2);
blurred = imgaussfilt(img,5/(2*sqrt(2*log(2))));
plot(blurred(zoomInLocInOrigPix(2),cropExt))

receptorColors = copper(numReceptors);
ii = 1;
hold on;
for receptorCenter = 0.25:5:180
    if receptorCenter > receptorStartLocDeg-90 && receptorCenter < receptorStartLocDeg-90 + numReceptors*5
        thisColor = receptorColors(ii,:);
        ii = ii + 1;
    else
        thisColor = [0.7 0.7 0.7];
    end
    plot(receptorCenter/pixelSizeDeg,-0.25,'.','Color',thisColor,'MarkerSize',20)
end
% set(gca,'visible','off')
set(gca,'xtick',[])
hold off;

%% get velocities
rng(5);
halfLife = 0.2; % s
velStd = 100; % degrees/s
sampleFreq = 100; % Hz
totalTime = 5; % s

numTime = sampleFreq*totalTime+1;

tau = halfLife/log(2);
filterBufferTime = 10*halfLife;
filterBufferNum = filterBufferTime*sampleFreq;

% assume velocity is normally distributed with a std of 100 in a
% natural setting. This is based on fly turning rates
velNoFilter = velStd*randn(filterBufferNum+numTime,1);

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

figure(3);
clf;
hold on;
plot(vel);
plot([1 length(vel)],[0 0],'k');
axis tight;
ylim([-300 300])
yticks([-300 300]);
plot([0, 50],[100 100]);
% set(gca,'xtick',[])
hold off;

%%
receptorOffsetsDeg = (0:5:45) + zoomInLocInOrigPix(1)*pixelSizeDeg;
receptorLocs = (mod(repmat(position,1,10)+receptorOffsetsDeg,360))/pixelSizeDeg;
receptorVals = blurred(zoomInLocInOrigPix(2),floor(receptorLocs)+1);
receptorVals = reshape(receptorVals,size(receptorLocs));

figure(4);
clf;
receptorValsSeparated = receptorVals + (1:10)/10;
hold on;
for ii = 1:numReceptors
    plot(receptorValsSeparated(:,ii),'Color',receptorColors(ii,:));
end
set(gca,'visible','off')
set(gca,'xtick',[])
hold off;

% figure(5);
% plot(receptorLocs(:,1)-cropExt(1));

% figure(6);
% plot(receptorVals(:,1));
