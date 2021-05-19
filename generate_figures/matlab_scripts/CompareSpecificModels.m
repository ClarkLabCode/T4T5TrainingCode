%% Find the top performers in each category

mpath = mfilename('fullpath');

figureWidth = 300;
figureHeight = 200;
figureHeightOffset = figureHeight + 60;

%% Load images
xtPlotFolderRelPath = [mpath '/../../../data_sets'];
listing = dir(xtPlotFolderRelPath);
xtPlotFolder = [listing(1).folder '/'];
if ~exist('images','var')
    testScenePaths = {[xtPlotFolder 'xtPlot_ns241_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-25_no0_csInf_sy0.mat'],...
                      [xtPlotFolder 'xtPlot_sineWaves_sl20_ll90_pe360_ps5_sf100_tt1_nt6080_hl0-2_vs100_df0-05_no0_csInf.mat'],...
                      };
    images = load(testScenePaths{1});
end

%%
outputNames = {'LN_tanh','LN_nonsym','LN_highestnoise','LNLN_highestnoise','conductance_highestnoise','sine_highestnoise','direction_highestnoise','LN_lownoise','LNLN_lownoise'};
runFolders = {'LNTanH','LN_nonsym','NoiseAndModelSweep','NoiseAndModelSweep','NoiseAndModelSweep','Sine','Direction','NoiseAndModelSweep','NoiseAndModelSweep'};
noiseLevels = [1,1,1,1,1,1,1,0.125,0.125];
modelNames = {'ln_model_flip','ln_model_flip','ln_model_flip','lnln_model_flip','conductance_model_flip','ln_model_flip','ln_model_flip','ln_model_flip','lnln_model_flip'};
for paramIdx = 1:length(outputNames)
    params = {};
    matfiles = dir([mpath '/../../../training_outputs/' runFolders{paramIdx} '/*.mat']);
    for xx = 1:length(matfiles)
        thisParams = load(fullfile(matfiles(xx).folder,matfiles(xx).name));
        params = cat(2,params,thisParams.param_dict);
    end
    chosen_model = GetModel(params,'max','model_name',modelNames{paramIdx},'input_noise_std',noiseLevels(paramIdx),'output_noise_std',noiseLevels(paramIdx));
    
    chosen_model.model_name = chosen_model.model_function_name;

    [h, b1, m1, m2, b2, model_structure] = assignModelParams(chosen_model);
    if any(isnan(h(:)))
        continue
    end
    num_filt = size(h,3);
    t_sample_rate = double(chosen_model.sample_freq);
    x_step = double(chosen_model.phase_step);

    %% New way to assign filter order
    newOrder = [1:4];
    if strcmp(outputNames{paramIdx},'LN_nonsym')
        newOrder = [1 2 4 3];
    end
    for jj = 1:2
        if (m2(jj) < m2(jj+2))
            newOrder([jj,jj+2]) = newOrder([jj+2,jj]);
        end
    end

    flashDataset = zeros(59,3,2);
    flashDataset(30:end,:,1) = 1;
    flashDataset(30:end,:,2) = -1;
    [~, component_flashResps] = model_structure(flashDataset,h,b1,m1,b2,m2);
    flashResps = cat(2,component_flashResps{1:2});
    peakFlashResps = squeeze(max(abs(flashResps)));
    differentialFlashResps = peakFlashResps(:,1) - peakFlashResps(:,2);

    if differentialFlashResps(2) > differentialFlashResps(1)
        newOrder = newOrder([2 1 4 3]);
    end

    h = h(:,:,newOrder);
    b1 = b1(:,newOrder);
    m1 = m1(:,newOrder);
    m2 = m2(newOrder);
    b2 = b2(newOrder);
    
    %% Plot Filter
%     fh = figure(1 + (paramIdx-1)*5);
%     pause(0.1);
%     clf;
%     
%     t = (0:size(h,1)-1)/t_sample_rate*1000;
%     t_label = linspace(0,(size(h,1)-1)/t_sample_rate*1000,4);
%     x = linspace(0,(size(h,2)-1)*x_step,size(h,2));
% 
%     for pp = 1:num_filt/2
%         scaledH(:,:,pp) = h(:,:,pp).*m1(:,pp)';
%     end
% 
%     if strcmp(chosen_model.model_name,'ln_model_flip') %ln model use proper convolution in space, other models use xcorr
%         scaledH = fliplr(scaledH);
%         flippedM1 = flipud(m1);
%     else
%         flippedM1 = m1;
%     end
% 
%     c_max = max(abs(scaledH(:)));
% 
%     for pp = 1:num_filt/2
%         %We're plotting input noise as we go up and down, weight noise
%         %horizontally
%         subplot(1,2,pp);
%         pause(0.1);
%         imagesc(x, t_label, scaledH(:,:,pp));
%         caxis([-c_max c_max]);
%         colormap(flipud(cbrewer('div','RdBu',100)));
% 
%         signChars = {'▁╱','╱▔'};%{'_/','/-'};
%         signStrs = arrayfun(@(x) signChars{(-sign(x)+1)/2 + 1},flippedM1(:,pp),'UniformOutput',false);
%         signLabel = ['  ' signStrs{1} '  ' signStrs{2} '  ' signStrs{3} '  '];
%         xlabel(signLabel,'Interpreter','none');
%     end
%     set(fh, 'Position', [figureWidth*(paramIdx-1) figureHeightOffset*4 figureWidth figureHeight])
%     
%     filterShapesOut.x = x;
%     filterShapesOut.t = t;
%     filterShapesOut.filterShapes = scaledH(:,:,1:2);
%     filterShapesOut.componentSigns = sign(flippedM1(:,1:2));
    
    
    fh = figure(1 + (paramIdx-1)*5);
    pause(0.1);
    clf;
    
    t = (0:size(h,1)-1)/t_sample_rate*1000;
    t_label = linspace(0,(size(h,1)-1)/t_sample_rate*1000,4);
    x = linspace(0,(size(h,2)-1)*x_step,size(h,2));

    if strcmp(chosen_model.model_name,'conductance_model_flip')
        randInput = randn(30,3,1000000);
        [~, component_rand_ouput] = model_structure(randInput,h,b1,m1,b2,m2);
        for pp = 1:2
            output = component_rand_ouput{pp};
            msOutput = output(:) - mean(output(:));
            flattenedInput = reshape(randInput,90,[]);
            extractedFilter = regress(msOutput,flattenedInput');
            extractedFilter = reshape(extractedFilter,30,3);
            scaledH(:,:,pp) = flipud(extractedFilter) ./ max(abs(extractedFilter),[],1);
            c_max = 1;
            subplot(1,2,pp);
            pause(0.1);
            imagesc(x, t_label, scaledH(:,:,pp));
            caxis([-c_max c_max]);
            colormap(flipud(cbrewer('div','RdBu',100)));
        end
    else
        for pp = 1:num_filt/2
            scaledH(:,:,pp) = h(:,:,pp).*m1(:,pp)';
        end

        if any(strcmp(chosen_model.model_name,{'ln_model_flip','ln_tanh_model_flip','ln_model'})) %ln model use proper convolution in space, other models use xcorr
            scaledH = fliplr(scaledH);
            flippedM1 = flipud(m1);
        else
            flippedM1 = m1;
        end

        scaledH = scaledH ./ max(abs(scaledH),[],1);

        c_max = max(abs(scaledH(:)));

        for pp = 1:num_filt/2
            %We're plotting input noise as we go up and down, weight noise
            %horizontally
            subplot(1,2,pp);
            pause(0.1);
            imagesc(x, t_label, scaledH(:,:,pp));
            caxis([-c_max c_max]);
            colormap(flipud(cbrewer('div','RdBu',100)));

            signChars = {'▁╱','╱▔'};%{'_/','/-'};
            signStrs = arrayfun(@(x) signChars{(-sign(x)+1)/2 + 1},flippedM1(:,pp),'UniformOutput',false);
            signLabel = ['  ' signStrs{1} '  ' signStrs{2} '  ' signStrs{3} '  '];
            xlabel(signLabel,'Interpreter','none');
        end
    end
    set(fh, 'Position', [figureWidth*(paramIdx-1) figureHeightOffset*4 figureWidth figureHeight])
    
    filterShapesOut.x = x;
    filterShapesOut.t = t;
    filterShapesOut.filterShapes = scaledH(:,:,1:2);
    filterShapesOut.componentSigns = sign(flippedM1(:,1:2));
    

    
    %% Responses to edges
    pause(0.1);
    fh = figure(2 + (paramIdx-1)*5);
    pause(0.1);
    clf;
    
    edgesDataset = GenerateSimpleEdgeDataset();
    [model_edge_output, component_edge_output] = model_structure(edgesDataset,h,b1,m1,b2,m2);
    numComponents = 4;%length(component_edge_output);
    maxAbs = 0;
    for compIdx = 1:numComponents
        for edgeIdx = 1:4
            thisMaxAbs = max(abs(component_edge_output{compIdx}(:,24,edgeIdx)));
            if thisMaxAbs > maxAbs
                maxAbs = thisMaxAbs;
            end
        end
    end
    if maxAbs == 0 % Make sure we set sensible plot limits
        maxAbs = 1;
    end

    edgeStrs = {'🡺 light','🡺 dark','🡸 light','🡸 dark'};
    for edgeIdx = 1:4
        for compIdx = 1:numComponents/2
            subplot(numComponents/2,4,(compIdx-1)*4 + edgeIdx,'replace');
            pause(0.01);
            plot(component_edge_output{compIdx}(:,24,edgeIdx))
            if compIdx == 1
                title(edgeStrs{edgeIdx});
            end
            if edgeIdx == 1
                ylabel(['C' num2str(compIdx)]);
            end
            ylim([-maxAbs,maxAbs]);
            xticks([]);
            yticks([]);
        end
    end
    set(fh, 'Position', [figureWidth*(paramIdx-1) figureHeightOffset*3 figureWidth figureHeight])
    
    respLength = size(component_edge_output{1},1);
    t = (0:respLength-1)/t_sample_rate*1000;
    movingEdgeRespsOut.t = t;
    movingEdgeRespsOut.resps = cat(3,squeeze(component_edge_output{1}(:,24,:)),squeeze(component_edge_output{2}(:,24,:)));
    
    
    componentsConcat = cat(4,component_edge_output{:});
    maxAbsResps = squeeze(max(abs(componentsConcat(:,36,:,:))));
    meanPDMaxAbsResp = mean(maxAbsResps(1:2,1:2),1); % Mean for each component across the two PD edges
    meanNDMaxAbsResp = mean(maxAbsResps(3:4,1:2),1);
    meanLightMaxAbsResp = mean(maxAbsResps([1 3],1:2),1);
    meanDarkMaxAbsResp = mean(maxAbsResps([2 4],1:2),1);

    esiPerComponent = (meanLightMaxAbsResp - meanDarkMaxAbsResp)./(meanLightMaxAbsResp + meanDarkMaxAbsResp +eps);
    esi = mean(abs(esiPerComponent),2);

    dsiPerComponent = (meanPDMaxAbsResp - meanNDMaxAbsResp)./(meanPDMaxAbsResp + meanNDMaxAbsResp +eps);
    dsi = mean(abs(dsiPerComponent),2);
    %% Static edges
    xLocs = (-5:5:165) - 0.0001;
    bars = square((2*pi/80) * xLocs);
%     
%     staticEdgeStimulus = repmat(bars,[1000,1]);
%     
    staticEdgeStimulus = GenerateStaticEdgeDataset();
    
    [model_static_edge_output, component_static_edge_output] = model_structure(staticEdgeStimulus,h,b1,m1,b2,m2);
    
    fh = figure(3 + (paramIdx-1)*5);
    pause(0.1);
    clf;
    hold all;
    allComponentTimeAveragedEdgeResps = cellfun(@(X) mean(X,1)', component_static_edge_output,'UniformOutput',false);
    allComponentTimeAveragedEdgeResps = cat(2,allComponentTimeAveragedEdgeResps{:});
    plot(xLocs(2:end-1),staticEdgeStimulus(1,2:end-1)*max(mean(component_static_edge_output{1},1)));
    plot(xLocs(2:end-1),allComponentTimeAveragedEdgeResps);
    plot(xLocs(2:end-1),bars(2:end-1)*max(mean(component_static_edge_output{1},1)));
    hold off;

    set(fh, 'Position', [figureWidth*(paramIdx-1) figureHeightOffset*2 figureWidth figureHeight])
    
    staticEdgeRespsOut.bars = bars(2:end-1);
    staticEdgeRespsOut.resps = allComponentTimeAveragedEdgeResps;
    staticEdgeRespsOut.respLocs = xLocs(2:end-1);
    %% Opponency testing
    fh = figure(4 + (paramIdx-1)*5);
    pause(0.1);
    clf;
    hUpscaled = interp1(1:30,h,linspace(1,30,300));
    sineDataset = GenerateSineDataset([60,-60],60,2)*GetSineScaleFactor(60);
    sineDataset = 0.5*sineDataset;
    sineDataset = cat(3,sineDataset,sum(sineDataset,3));

    [model_sine_output, component_sine_output] = model_structure(sineDataset,hUpscaled,b1,m1,b2,m2);

    modelAvResp = mean(model_sine_output,[1 2]);
    
    componentAvResps = zeros(3,2);
    for cc = 1:2
        componentAvResps(:,cc) = mean(component_sine_output{cc},[1 2]);
    end

    combinedResps = [modelAvResp(:), componentAvResps];
    bar(combinedResps');
    xticklabels({'full','C1','C2'});
    legend({'L','R','L+R'},'Location','northwestoutside');
    set(fh, 'Position', [figureWidth*(paramIdx-1) figureHeightOffset*1 figureWidth figureHeight])
    
    opponencyRespsOut.resps = componentAvResps;
    %% Re-evaluate on natural secese
    testSceneIdx = 1;

    % switch dev to test set if you want
    data_set = images.dev_in;
    data_ans = images.dev_out;

    % set up data_set
    data_set = permute(data_set, [2 1 3]);

    data_set = (data_set - mean(data_set,[1 2]))./std(data_set,[],[1 2]);

    % get model output
    [model_output, component_output] = model_structure(data_set,h,b1,m1,b2,m2);
    %% Coactivation
    co_act = zeros(num_filt);
    for aa = 1:num_filt
        for bb = 1:num_filt
            ms_a = abs(component_output{aa});% - mean(abs(component_output{aa}(:)));
            ms_b = abs(component_output{bb});% - mean(abs(component_output{aa}(:)));

            rms_a = rms(ms_a, [1 2]);
            rms_b = rms(ms_b, [1 2]);

            scaled_a = (ms_a+eps)./(rms_a + eps);
            scaled_b = (ms_b+eps)./(rms_b + eps);
            co_act(aa,bb) = mean(scaled_a.*scaled_b, [1, 2, 3]);
        end
    end
    fh = figure(5 + (paramIdx-1)*5);
    pause(0.1);
    clf;
    aye = eye(4);
    if abs(co_act(:)' * aye(:) - 4) > 0.01
        keyboard
    end
    sparsity = (0.8660 - sqrt(sum((co_act - eye(4)).^2, [1,2])/16))/0.8660;
    imagesc(abs(co_act))
    actLim = max(abs(co_act(:)));
    if ~isnan(actLim)
        caxis([-actLim actLim]);
    end
    colormap(flipud(cbrewer('div','RdBu',100)));
    title(['Coactivation, sparsity = ' num2str(sparsity,2)]);
    set(fh, 'Position', [figureWidth*(paramIdx-1) figureHeightOffset*0 figureWidth figureHeight])
    
    coactivationOutput.resps = co_act;
    %%
    outRelPath = [mpath '/../../intermediate_mat_files/'];
    listing = dir(outRelPath);
    outpath = [listing(1).folder '/'];
    save([outpath outputNames{paramIdx} '.mat'],'filterShapesOut','movingEdgeRespsOut','staticEdgeRespsOut','opponencyRespsOut','coactivationOutput');
end
