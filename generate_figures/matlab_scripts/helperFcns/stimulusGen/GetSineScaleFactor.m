function factor = GetSineScaleFactor(lambda)

x = (0:0.1:359.9)';
wave = sind(x * 360/lambda)';

filtStd = 5/(2*sqrt(2*log(2))); % 5 degree FWHM

xFilt = ifftshift(normpdf(x,180,filtStd))'/10;

fftScene = fft(wave);
fftFilt = fft(xFilt);

filtered = real(ifft(fftScene.*fftFilt));

factor = max(filtered);