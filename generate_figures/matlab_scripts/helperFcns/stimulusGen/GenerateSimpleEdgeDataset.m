function edgesContrast = GenerateSimpleEdgeDataset()

    x = (0:1:359)';    
    
    % individual photo receptors
    filtStd = 5/(2*sqrt(2*log(2))); % 5 degree FWHM

    xFilt = ifftshift(normpdf(x,180,filtStd))';
    
    % perform the convolution
    vel = 30;
    totalTime = 360/vel;
    tRes = 0.01;
    numTimesteps = totalTime/tRes;
    tts = 0:tRes:totalTime-tRes;
    for tIdx = 1:numTimesteps
        epsilon = 0.0001;
        edge = -1*ones(1,360);%heaviside(linspace(-1,1,360))*(1-epsilon) + epsilon;
        pos = floor(vel*tts(tIdx))+1;
        edge(1:pos) = 1;

        fftScene = fft(edge);
        fftFilt = fft(xFilt);

        edgeContrast(tIdx,:) = real(ifft(fftScene.*fftFilt));
    end
    edgeContrast = edgeContrast(:,60:5:300);
    edgesContrast = cat(3,edgeContrast,rot90(edgeContrast,2),fliplr(edgeContrast),flipud(edgeContrast));

%     MakeFigure;
%     for ii = 1:4
%         subplot(4,1,ii)
%         imagesc(edgesContrast(:,:,ii));
%     end
    
    
end