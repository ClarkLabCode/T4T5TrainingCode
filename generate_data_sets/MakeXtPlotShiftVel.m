function xtPlot = MakeXtPlotShiftVel(xyPlot,startX,startY,position,phi)
    % xyPlot is a matrix of natural scenes that is y by x by scenes
    % startX is a vector in deg
    % startY is a vector in deg
    % position is a matrix of imagePosition, xy points by time
    % sample freq is in hz
    % totalTime is the amount of time to simulate
    % phi is a vector of the offset of your visual inputs
    
    numT = size(position,1);
    
    % shift the image over time at velocity vel to make an xt plot
    numSamples = size(startX,1);
    numTraces = size(position,3);
    numScenes = size(xyPlot,3);

    xyRes = 360/size(xyPlot,2); 
    yPointsInd = ceil(startY/xyRes);

    % add one extra value so you can interpolate circularly
    xyPlot = [xyPlot xyPlot(:,1,:)];
    x = (0:xyRes:360)';
    
    numPhi = length(phi);
    xtPlot = zeros(numT,numPhi,numScenes,numSamples,numTraces);
    
    parfor ss = 1:numSamples
%         samTimeStart = tic;

        for sc = 1:numScenes
            for pp = 1:numPhi
                posMod = mod(-position(:,ss,:)+startX(ss)+phi(pp),360);
                xtPlot(:,pp,sc,ss,:) = interp1(x,xyPlot(yPointsInd(ss),:,sc),posMod);
            end
        end
        
%         samTimeEnd = toc(samTimeStart);
%         
%         if mod(ss,round(numSamples/10))==0
%             disp([num2str(ss) '/' num2str(numSamples) ' xt plots made and took ' num2str(samTimeEnd) ' seconds per plot']);
%             disp([num2str(samTimeEnd*(numSamples-ss)) ' seconds remaining']);
%         end
    end
end