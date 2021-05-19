function edgesContrast = GenerateStaticEdgeDataset()

    x = linspace(0,360,10000);
    x = x(1:end-1);
    midpointIdx = 1+(length(x)-1)/2;
    
    % individual photo receptors
    filtStd = 5/(2*sqrt(2*log(2))); % 5 degree FWHM

    xFilt = ifftshift(normpdf(x,x(midpointIdx),filtStd))'*x(2); %normalize by spacing so filtering doesn't change magnitude
    
    scene = square((2*pi/80) * (x-10 - 0.0001))';

    fftScene = fft(scene);
    fftFilt = fft(xFilt);

    edgesContrastHighRes = real(ifft(fftScene.*fftFilt));
    
    samplePoints = (2.5:5:172.5)+80;
    edgesContrastDownsampled = interp1(x,edgesContrastHighRes,samplePoints);
    edgesContrast = repmat(edgesContrastDownsampled,[1000,1]);
    
    figure(1);
    clf;
    hold on;
    xLocs = (-5:5:165) - 0.0001;
    bars = square((2*pi/80) * xLocs);
    plot(bars);
    plot(bars,'.');
    plot(edgesContrastDownsampled)
    
    
end