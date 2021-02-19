function edgesContrast = GenerateEdgeDataset(imageMeanContrast)
    %% load images
    if nargin < 1
        imageMeanContrast = 0;
    end
        
    x = (0:1:359)';    
    
    % individual photo receptors
    filtStd = 5/(2*sqrt(2*log(2))); % 5 degree FWHM

    xFilt = ifftshift(normpdf(x,180,filtStd))';
    
    % for mean estimation
    filtStdContrast = 25/(2*sqrt(2*log(2)));

    xFiltContrast = ifftshift(normpdf(x,180,filtStdContrast))';
    
    % perform the convolution
    vel = 40;
    totalTime = 360/vel;
    tRes = 0.01;
    numTimesteps = totalTime/tRes;
    tts = 0:tRes:totalTime-tRes;
    for tIdx = 1:numTimesteps
        epsilon = 0.0001;
        edge = zeros(1,360);%heaviside(linspace(-1,1,360))*(1-epsilon) + epsilon;
        pos = floor(vel*tts(tIdx))+1;
        edge(1:pos) = 1;
        edge = edge*(1-epsilon)+epsilon;

        fftScene = fft(edge);
        fftFilt = fft(xFilt);
        fftFiltContrast = fft(xFiltContrast);

        filteredScene = real(ifft(fftScene.*fftFilt));
        filteredSceneContrast = real(ifft(fftScene.*fftFiltContrast));

        if imageMeanContrast
            imageMean = mean(filteredScene(:));
            edgeContrast(tIdx,:) = (filteredScene-imageMean)./imageMean;
        else
            edgeContrast(tIdx,:) = (filteredScene-filteredSceneContrast)./filteredSceneContrast;
        end
    end
    edgeContrast = edgeContrast(:,60:5:300);
    edgesContrast = cat(3,edgeContrast,rot90(edgeContrast,2),fliplr(edgeContrast),flipud(edgeContrast));
%     MakeFigure;
%     subplot(4,1,1);
%     plot(edge)
%     subplot(4,1,2);
%     plot(filteredScene)
%     subplot(4,1,3);
%     plot(filteredSceneContrast)
%     subplot(4,1,4);
%     plot(edgesContrast);

%     MakeFigure;
%     for ii = 1:4
%         subplot(4,1,ii)
%         imagesc(edgesContrast(:,:,ii));
%     end
%     
    
end