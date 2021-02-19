function NaturalScenesToContrastImageMean
    %% load images
    scenes = [];
    matfiles = dir('pano_scenes/*.mat');
    for ii = 1:length(matfiles)
        data = load(fullfile('pano_scenes',matfiles(ii).name));
        scenes = cat(3,scenes,data.projection);
    end

    xRes = 360/size(scenes,2);
    
    x = (0:xRes:360-xRes)';
    y = (0:xRes:xRes*size(scenes,1)-xRes)';
    
    
    % individual photo receptors
    filtStd = 5/(2*sqrt(2*log(2))); % 5 degree FWHM

    xFilt = ifftshift(normpdf(x,180,filtStd));
    yFilt = ifftshift(normpdf(y,(y(end)+xRes)/2,filtStd));
    
    xyFiltMat = yFilt*xFilt';
    xyFiltMat = xyFiltMat/sum(xyFiltMat(:));
    
    % perform the convolution
    xyPlot = zeros(size(scenes));
    for ii = 1:size(scenes,3)
        fftScene = fft2(scenes(:,:,ii));
        fftFilt = fft2(xyFiltMat);
        filteredScene = real(ifft2(fftScene.*fftFilt));

        filteredSceneMean = mean(filteredScene(:));
        xyPlot(:,:,ii) = (filteredScene-filteredSceneMean)./filteredSceneMean;
    end
    
    save('combinedFiltered2DImageMeanContrast.mat','xyPlot');
end