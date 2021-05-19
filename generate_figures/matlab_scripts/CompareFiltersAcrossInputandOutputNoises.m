quickRun = false;


modelNames = {'ln_model','lnln_model','conductance_model'};
params = {};
mpath = mfilename('fullpath');
matfiles = dir([mpath '/../../../training_outputs/NoiseAndModelSweep/*.mat']);
for xx = 1:length(matfiles)
    thisParams = load(fullfile(matfiles(xx).folder,matfiles(xx).name));
    params = cat(2,params,thisParams.param_dict);
end


xtPlotFolderRelPath = [mpath '/../../../data_sets'];
listing = dir(xtPlotFolderRelPath);
xtPlotFolder = [listing(1).folder '/'];
if ~exist('images','var')
    testScenePaths = {[xtPlotFolder 'xtPlot_ns241_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-25_no0_csInf_sy0.mat'],...
                      [xtPlotFolder 'xtPlot_sineWaves_sl20_ll90_pe360_ps5_sf100_tt1_nt6080_hl0-2_vs100_df0-05_no0_csInf.mat'],...
                      };
    images = load(testScenePaths{1});
    for testSceneIdx = 2:length(testScenePaths)
        images(testSceneIdx) = load(testScenePaths{testSceneIdx});
    end
end

noise_range = [0, 0.125, 0.25, 0.5, 1.0];
num_noises = length(noise_range);

for ff = 1:3
    figure(ff);
    clf;
end

for model_idx = 1:3
    for input_noise_idx = 1:num_noises
        for weight_noise_std_idx = 1:num_noises
            for modelRank = 1:9
                input_noise = noise_range(input_noise_idx);
                weight_noise_std = noise_range(weight_noise_std_idx);
                modelName = [modelNames{model_idx} '_flip'];
                sumOverSpace = 0;
                normalizeStd = 1;
                sceneType = params{1}.image_type;
%                 chosen_models = GetModel(params,'all','model_name',modelName,'sum_over_space',sumOverSpace,'normalize_std',normalizeStd,'image_type',sceneType,'filter_weight_noise_std',weight_noise_std,'noise_std',input_noise);%, 'use_activity_regularizer', used_activity_regularizer);
                chosen_models = GetModel(params,'all','model_name',modelName,'image_type',sceneType,'output_noise_std',weight_noise_std,'input_noise_std',input_noise);%, 'use_activity_regularizer', used_activity_regularizer);
                chosen_model = chosen_models(end-(modelRank-1));
                chosen_model.model_name = modelName;

                [h, b1, m1, m2, b2, model_structure] = assignModelParams(chosen_model);
                if any(isnan(h(:)))
                    continue
                end
                num_filt = size(h,3);
                t_sample_rate = double(chosen_model.sample_freq);
                x_step = double(chosen_model.phase_step);

                %% New way to assign filter order
                newOrder = [1:4];
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

                %% Re-evaluate on natural secese
                testSceneIdx = 1;

                % switch dev to test set if you want
                data_set = images(testSceneIdx).dev_in;
                data_ans = images(testSceneIdx).dev_out;
                
                if quickRun
                    % Use first 5 of 61 screens to evaluate
                    data_set = data_set(:,:,1:2*19*5);
                    data_ans = data_ans(:,1:2*19*5);
                end
                
                % set up data_set
                data_set = permute(data_set, [2 1 3]);

%                 if chosen_model.normalize_std
                    data_set = (data_set - mean(data_set,[1 2]))./std(data_set,[],[1 2]);
%                 end

                % get model output
                if model_idx == 3
                    [model_output, component_output, numerators, denominators] = model_structure(data_set,h,b1,m1,b2,m2);
                    meanDenominator(modelRank,input_noise_idx,weight_noise_std_idx) = mean(cat(4,denominators{:}),'all');
                else
                    [model_output, component_output] = model_structure(data_set,h,b1,m1,b2,m2);
                end

                %Calculate with low noise
                noisy_data_set = data_set + randn(size(data_set))*0.125;
                [~, noisy_component_output] = model_structure(noisy_data_set,h,b1,m1,b2,m2);
                mean_val = 1;
                var = 0.125^2;
                mu = log((mean_val^2)/sqrt(var + mean_val^2));
                sig = sqrt(log(var/(mean_val^2) + 1));
                output_size = size(noisy_component_output{1});
                noisy_model_output = zeros(output_size);
                for ii = 1:4
                    noisy_model_output = noisy_model_output + noisy_component_output{ii} .* lognrnd(mu,sig,output_size);
                end
                
                num_filt = length(component_output);

                % set up data_ans
                data_ans = repmat(permute(data_ans, [1 3 2]), [1 size(component_output{1}, 2) 1]);
                data_ans = data_ans(end-size(component_output{1},1)+1:end, :, :);

                % R2
                trainingR2 = chosen_model.r2(end);
                num = sum((data_ans(:)-noisy_model_output(:)).^2);
                denom = sum((data_ans(:)-mean(data_ans(:))).^2);
                varExpl(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = 1-mean(num./denom);

                r = corrcoef(data_ans(:),noisy_model_output(:));
                r2(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = r(2)^2;

                sum_model_output = mean(noisy_model_output,2);
                sum_data_ans = data_ans(:,1,:);
                num = sum((sum_data_ans(:)-sum_model_output(:)).^2);
                denom = sum((sum_data_ans(:)-mean(sum_data_ans(:))).^2);
                sumVarExpl(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = 1-mean(num./denom);

                
                %Calculate with high noise
                noisy_data_set = data_set + randn(size(data_set))*1;
                [~, noisy_component_output] = model_structure(noisy_data_set,h,b1,m1,b2,m2);
                mean_val = 1;
                var = 1;
                mu = log((mean_val^2)/sqrt(var + mean_val^2));
                sig = sqrt(log(var/(mean_val^2) + 1));
                output_size = size(noisy_component_output{1});
                noisy_model_output = zeros(output_size);
                for ii = 1:4
                    noisy_model_output = noisy_model_output + noisy_component_output{ii} .* lognrnd(mu,sig,output_size);
                end
                
                num = sum((data_ans(:)-noisy_model_output(:)).^2);
                denom = sum((data_ans(:)-mean(data_ans(:))).^2);
                varExplNoisy(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = 1-mean(num./denom);

                r = corrcoef(data_ans(:),noisy_model_output(:));
                r2Noisy(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = r(2)^2;

                sum_model_output = mean(noisy_model_output,2);
                sum_data_ans = data_ans(:,1,:);
                num = sum((sum_data_ans(:)-sum_model_output(:)).^2);
                denom = sum((sum_data_ans(:)-mean(sum_data_ans(:))).^2);
                sumVarExplNoisy(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = 1-mean(num./denom);
                
%                 % percent on
%                 if testSceneIdx == 1
%                     channels = zeros(size(data_set,1)-size(h,1)+1,size(data_set,2)-2,size(data_set,3),3,num_filt);
%                     for cc = 1:num_filt
%                         percentNonZero(cc) = 100*mean((component_output{cc}(:) ~= 0));
% 
%                         % convolve each filter in the model with each image in the dev set
%                         for ff = 1:size(data_set,3)
%                             for xx = 1:3
%                                 channels(:,:,ff,xx,cc) = conv2(data_set(:,xx:end-3+xx,ff), h(:,xx,cc), 'valid') + b1(xx, cc);
%                             end
%                         end
%                     end
%                     channelPercentNonZero = squeeze(100*mean(channels>0,[1 2 3]));
%                 end


                % calculate coactivation of the filters for natural scenes (first
                % one)

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
                sparsity(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = (0.8660 - sqrt(sum((co_act - eye(4)).^2, [1,2])/16))/0.8660;

                %% Responses to moving edges
                edgesDataset = GenerateSimpleEdgeDataset();
                if model_idx == 3
                    [model_edge_output, component_edge_output, numerators, denominators] = model_structure(edgesDataset,h,b1,m1,b2,m2);
                else
                    [model_edge_output, component_edge_output] = model_structure(edgesDataset,h,b1,m1,b2,m2);
                end
                numComponents = 4;%length(component_edge_output);

                componentsConcat = cat(4,component_edge_output{:});
                maxAbsResps = squeeze(max(abs(componentsConcat(:,24,:,:))));
                meanPDMaxAbsResp = mean(maxAbsResps(1:2,1:2),1); % Mean for each component across the two PD edges
                meanNDMaxAbsResp = mean(maxAbsResps(3:4,1:2),1);
                meanLightMaxAbsResp = mean(maxAbsResps([1 3],1:2),1);
                meanDarkMaxAbsResp = mean(maxAbsResps([2 4],1:2),1);

                esiPerComponent = (meanLightMaxAbsResp - meanDarkMaxAbsResp)./(meanLightMaxAbsResp + meanDarkMaxAbsResp +eps);
                esi(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = mean(abs(esiPerComponent),2);

                dsiPerComponent = (meanPDMaxAbsResp - meanNDMaxAbsResp)./(meanPDMaxAbsResp + meanNDMaxAbsResp +eps);
                dsi(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = mean(abs(dsiPerComponent),2);

%                 if model_idx == 3
%                     for filtNum = 1:2
%                         
%                         for epoch = 1:4
%                             [maxResps(epoch),maxRespIdx(epoch)] = max(component_edge_output{filtNum}(:,24,epoch));
%                         end
%                         [~,prefEpoch] = max(maxResps(1:2));
%                         nullEpoch = 5-prefEpoch;
%                         otherEpochs = [3 - prefEpoch, 2 + prefEpoch];
%                         %would like to use this assert, but there are some
%                         %very wierd responses out there
%                         %assert(all(maxResps(nullEpoch) > maxResps(otherEpochs)));
%                         
%                         maxRespPrefIdx = maxRespIdx(prefEpoch);
%                         maxRespNullIdx = maxRespIdx(nullEpoch);
%                         numPrefResp = numerators{filtNum}(maxRespPrefIdx,24,prefEpoch);
%                         numNullResp = numerators{filtNum}(maxRespNullIdx,24,nullEpoch);
%                         thisNumeratorSelectivity(filtNum) = (numPrefResp - numNullResp)/(abs(numPrefResp) + abs(numNullResp));
%                         denPrefResp = denominators{filtNum}(maxRespPrefIdx,24,prefEpoch);
%                         denNullResp = denominators{filtNum}(maxRespNullIdx,24,nullEpoch);
%                         thisDenominatorSelectivity(filtNum) = (denPrefResp - denNullResp)/(abs(denPrefResp) + abs(denNullResp));
% %                         if thisNumeratorSelectivity(filtNum) - thisDenominatorSelectivity < 0
% %                             keyboard;
% %                         end
%                     end
%                     numeratorSelectivity(:,modelRank,input_noise_idx,weight_noise_std_idx) = thisNumeratorSelectivity;
%                     denominatorSelectivity(:,modelRank,input_noise_idx,weight_noise_std_idx) = thisDenominatorSelectivity;
%                 end
                %% Responses to static edges
                staticEdgesDataset = zeros(30,3,2);
                contrastVal = normcdf(sqrt(2*log(2))) - normcdf(-sqrt(2*log(2)));  % integral of receptive field 1 half max away
                staticEdgesDataset(:,1,1) = contrastVal;
                staticEdgesDataset(:,3,1) = -contrastVal;
                staticEdgesDataset(:,1,2) = -contrastVal;
                staticEdgesDataset(:,3,2) = contrastVal;
                [model_s_edge_output, component_s_edge_output] = model_structure(staticEdgesDataset,h,b1,m1,b2,m2);
                for cc = 1:2
                    edgeResponses = component_s_edge_output{cc}(:);
                    componentStaticEdgeActivation(cc) = mean(abs(edgeResponses));
                    componentStaticEdgeSelectivity(cc) = abs(edgeResponses(1) - edgeResponses(2))/sum(abs(edgeResponses));
                end
                staticEdgeActivation(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = mean(componentStaticEdgeActivation);
                staticEdgeSelectivity(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = mean(componentStaticEdgeSelectivity);
                %% Opponency testing
                sineDataset = GenerateSineDataset([60,-60],60,2)*GetSineScaleFactor(60);
                sineDataset = 0.5*sineDataset;
                orthogonalDirection = GenerateSineDataset([1e6],1e6,2)*0.5;
                orthoAdded = sineDataset+orthogonalDirection;
                sineDataset = cat(3,sineDataset,sum(sineDataset,3),orthoAdded);
                hUpscaled = interp1(1:30,h,linspace(1,30,300));
                [model_sine_output, component_sine_output] = model_structure(sineDataset,hUpscaled,b1,m1,b2,m2);
                modelAvResp = mean(model_sine_output,[1 2]);

                componentAvResps = zeros(5,2);
                for cc = 1:2
                    componentAvResps(:,cc) = mean(component_sine_output{cc},[1 2]);
                end

                [preferredResponse, prefDirIdx] = max(componentAvResps(1:2,:),[],1);
                for cc = 1:2
                    prefPlusOrthoResp(1,cc) = componentAvResps(3+prefDirIdx(cc),cc);
                end
                componentOpponencyIndex = (preferredResponse - componentAvResps(3,:))./(preferredResponse + componentAvResps(3,:) + eps) .* (preferredResponse ~= 0);
                opponencyIndex(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = mean(componentOpponencyIndex);
                odEnhancementIndex(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = mean((preferredResponse - prefPlusOrthoResp)./(preferredResponse + prefPlusOrthoResp),2);

%                 %% Plot Filters
%                 if modelRank == 1
%                     figure(model_idx);
%                     t = (0:size(h,1)-1)/t_sample_rate*1000;
%                     t_label = linspace(0,(size(h,1)-1)/t_sample_rate*1000,4);
%                     x = linspace(0,(size(h,2)-1)*x_step,size(h,2));
% 
%                     for pp = 1:num_filt/2
%                         scaledH(:,:,pp) = h(:,:,pp).*m1(:,pp)';
%                     end
%                     
%                     if strcmp(modelNames{model_idx},'ln_model') %ln model use proper convolution in space, other models use xcorr
%                         scaledH = fliplr(scaledH);
%                         flippedM1 = flipud(m1);
%                         flippedCPNZ = flipud(channelPercentNonZero);
%                     else
%                         flippedM1 = m1;
%                         flippedCPNZ = channelPercentNonZero;
%                     end
%                     
%                     c_max = max(abs(scaledH(:)));
% 
%                     for pp = 1:num_filt/2
%                         %We're plotting input noise as we go up and down, weight noise
%                         %horizontally
%                         subplot(num_noises,num_noises*num_filt/2,(input_noise_idx-1)*num_noises*2 + (weight_noise_std_idx-1)*2 + pp)
%                         imagesc(x, t_label, scaledH(:,:,pp));
%                         caxis([-c_max c_max]);
%                         colormap(flipud(cbrewer('div','RdBu',100)));
% 
%                         title([num2str(round(percentNonZero(pp)),'%2d') '%']);
% 
%                         signChars = {'▁╱','╱▔'};%{'_/','/-'};
%                         signStrs = arrayfun(@(x) signChars{(-sign(x)+1)/2 + 1},flippedM1(:,pp),'UniformOutput',false);
%                         percentStrs = arrayfun(@(X) sprintf('%2d%%',X), round(flippedCPNZ(:,pp)),'UniformOutput',false);
%                         percentLabel = ['  ' signStrs{1} percentStrs{1} '  ' signStrs{2} percentStrs{2} '  ' signStrs{3} percentStrs{3} '  '];
%                         xlabel(percentLabel,'Interpreter','none');
%                     end
%                 end
%                 %% Evaluate ST orientation
%                 for filtNum = 1:2
%                     scaledH(:,:,filtNum) = h(:,:,filtNum).*m1(:,filtNum)';
%                     if strcmp(modelNames{model_idx},'ln_model') %ln model use proper convolution in space, other models use xcorr
%                         scaledH = fliplr(scaledH);
%                     end
%                     FH = abs(fftshift(fft2(scaledH(1:29,:,filtNum)))); %Square?
%                     orientIdx(filtNum) = (sum(FH(1:14,1)) - sum(FH(1:14,3))) /(sum(sum(FH(1:14,:))) + FH(15,1) + 0*FH(15,2));
%                 end
%                 orientationIndex(modelRank,input_noise_idx,weight_noise_std_idx,model_idx) = mean(orientIdx);
            end
        end 
    end
end

modelNames = strrep(modelNames,'_',' ');
%% Draw lines between subplots
% for modelIdx = 1:3
%     figure(modelIdx);
%     xPositions = linspace(0.2747,0.7519,4);
%     yPositions = linspace(0.2485,0.7549,4);
%     for xx = 1:4
%         xloc = xPositions(xx);
%         yloc = yPositions(xx);
%         arh=annotation('line', [xloc,xloc], [0.12, 0.88]);
%         arh.LineWidth = 5;
%         arh=annotation('line', [0.12, 0.88], [yloc,yloc]);
%         arh.LineWidth = 5;
%     end
%     sgtitle(modelNames{modelIdx});
% end


%%    
figure(4);
clf;
for modelIdx = 1:3
    for plotTypeIdx = 1:3
        idx = (plotTypeIdx-1)*3 + modelIdx;
        subplot(3,3,idx);
        switch plotTypeIdx
            case 1            
                imagesc(convertTensorToPlotMatrix(varExpl(:,:,:,modelIdx)));
                title([modelNames{modelIdx} ' Variance Explained'])
            case 2
                imagesc(convertTensorToPlotMatrix(r2(:,:,:,modelIdx)));
                title([modelNames{modelIdx} ' R2']);
            case 3
                imagesc(convertTensorToPlotMatrix(sumVarExpl(:,:,:,modelIdx)))
                title('Variance Explained (Sum over space)');
        end
        ylabel('Input Noise');
        xlabel('Component Output Noise');
        xticks(1.5:2:num_noises*2);
        xticklabels(noise_range);
        yticks(1.5:2:num_noises*2);
        yticklabels(noise_range);
        colorbar;
        cl = caxis;
        if plotTypeIdx ~= 2
            caxis([0,cl(2)])
        end
    end
end

figure(5);
clf;
for modelIdx = 1:3
    for plotTypeIdx = 1:3
        idx = (plotTypeIdx-1)*3 + modelIdx;
        subplot(3,3,idx);
        switch plotTypeIdx
            case 1            
                imagesc(convertTensorToPlotMatrix(varExplNoisy(:,:,:,modelIdx)));
                title([modelNames{modelIdx} ' Variance Explained'])
            case 2
                imagesc(convertTensorToPlotMatrix(r2Noisy(:,:,:,modelIdx)));
                title([modelNames{modelIdx} ' R2']);
            case 3
                imagesc(convertTensorToPlotMatrix(sumVarExplNoisy(:,:,:,modelIdx)))
                title('Variance Explained (Sum over space)');
        end
        ylabel('Input Noise');
        xlabel('Component Output Noise');
        xticks(1.5:2:num_noises*2);
        xticklabels(noise_range);
        yticks(1.5:2:num_noises*2);
        yticklabels(noise_range);
        colorbar;
        cl = caxis;
        if plotTypeIdx ~= 2
            caxis([0,cl(2)])
        end
    end
end
sgtitle('Eval at high noise')

%%
figure(6);
clf;
for modelIdx = 1:3
    subplot(1,3,modelIdx)
    imagesc(convertTensorToPlotMatrix(sparsity(:,:,:,modelIdx)))
    ylabel('Input Noise');
    xlabel('Component Output Noise');
    xticks(1.5:2:num_noises*2);
    xticklabels(noise_range);
    yticks(1.5:2:num_noises*2);
    yticklabels(noise_range);
    colorbar;
    title(modelNames{modelIdx});
end
sgtitle('Sparsity');

figure(7);
for modelIdx = 1:3
    for xx = 1:2
        idx = (xx-1)*3 + modelIdx;
        subplot(2,3,idx);
        switch xx
            case 1            
                imagesc(convertTensorToPlotMatrix(esi(:,:,:,modelIdx)));
                title([modelNames{modelIdx} ' ESI'])
            case 2
                imagesc(convertTensorToPlotMatrix(dsi(:,:,:,modelIdx)));
                title([modelNames{modelIdx} ' DSI']);
        end
        ylabel('Input Noise');
        xlabel('Component Output Noise');
        xticks(1.5:2:num_noises*2);
        xticklabels(noise_range);
        yticks(1.5:2:num_noises*2);
        yticklabels(noise_range);
        colorbar
    end
end

figure(8);
clf;
for modelIdx = 1:3
    subplot(1,3,modelIdx);
    imagesc(convertTensorToPlotMatrix(opponencyIndex(:,:,:,modelIdx)))
    ylabel('Input Noise');
    xlabel('Component Output Noise');
    xticks(1.5:2:num_noises*2);
    xticklabels(noise_range);
    yticks(1.5:2:num_noises*2);
    yticklabels(noise_range);
    colorbar;
    title(modelNames{modelIdx});
end
sgtitle('Opponency');



% figure(9);
% clf;
% for modelIdx = 1:3
%     subplot(1,3,modelIdx);
%     imagesc(convertTensorToPlotMatrix(odEnhancementIndex(:,:,:,modelIdx)))
%     ylabel('Input Noise');
%     xlabel('Component Output Noise');
%     xticks(1.5:2:num_noises*2);
%     xticklabels(noise_range);
%     yticks(1.5:2:num_noises*2);
%     yticklabels(noise_range);
%     colorbar;
%     title(modelNames{modelIdx});
% end
% sgtitle('OD suppression index');
% 
% figure(10);
% clf;
% imagesc(convertTensorToPlotMatrix(meanDenominator))
% ylabel('Input Noise');
% xlabel('Component Output Noise');
% xticks(1.5:2:num_noises*2);
% xticklabels(noise_range);
% yticks(1.5:2:num_noises*2);
% yticklabels(noise_range);
% colorbar;
% title('Conductance Model Mean Denominator');
% 
% figure(11);
% clf;
% meanNumeratorSelectivity = squeeze(mean(numeratorSelectivity,1));
% meanDenominatorSelectivity = squeeze(mean(denominatorSelectivity,1));
% for ii = 1:2
%     subplot(2,1,ii);
%     if ii == 1
%         imagesc(convertTensorToPlotMatrix(meanNumeratorSelectivity))
%         title('Conductance Model Mean Numerator Selectivity');
%     else
%         imagesc(convertTensorToPlotMatrix(meanDenominatorSelectivity))
%         title('Conductance Model Mean Denominator Selectivity');
%     end
%     ylabel('Input Noise');
%     xlabel('Component Output Noise');
%     xticks(1.5:2:num_noises*2);
%     xticklabels(noise_range);
%     yticks(1.5:2:num_noises*2);
%     yticklabels(noise_range);
%     colorbar;
%     cl = caxis;
%     if cl(1) < -1
%         cl(1) = -1;
%     end
%     if cl(2) > 1
%         cl(2) = 1;
%     end
%     caxis(cl);
% end
% 
% figure(12);
% clf;
% subplot(1,2,1)
% a = linspace(-100,100,5);
% b = a;
% [A,B] = meshgrid(a,b);
% L = 75*ones(5);
% LAB = cat(3,L,A,B);
% rgb = lab2rgb(LAB);
% imagesc(rgb);
% axis image;
% ylabel('Input Noise');
% xlabel('Component Output Noise');
% xticks(1.5:2:num_noises*2);
% xticklabels(noise_range);
% yticks(1.5:2:num_noises*2);
% yticklabels(noise_range);
% 
% numModels = size(meanDenominatorSelectivity,1);
% reprgb = repmat(reshape(rgb,[1,5,5,3]),numModels*2,1,1);
% colormat = reshape(reprgb,[],3);
% subplot(1,2,2)
% markerSize = 6;
% hold on;
% plot([-0.5 0.5],[-0.5 0.5],'k--')
% scatter(denominatorSelectivity(:),numeratorSelectivity(:),markerSize,colormat,'filled')
% hold off;
% axis image
% ylabel('numerator selectivity')
% xlabel('denominator selectivity');
% xline(0,':','Color',[0.5,0.5,0.5]);
% yline(0,':','Color',[0.5,0.5,0.5]);
% 
% sgtitle('Conductance Model Num vs Den');
% 
% figure(13);
% clf;
% projection = numeratorSelectivity - denominatorSelectivity;
% meanProjection = squeeze(mean(projection,1));
% imagesc(convertTensorToPlotMatrix(meanProjection))
% title('Numeration - Denominator Selectivity');
% ylabel('Input Noise');
% xlabel('Component Output Noise');
% xticks(1.5:2:num_noises*2);
% xticklabels(noise_range);
% yticks(1.5:2:num_noises*2);
% yticklabels(noise_range);
% colorbar;

figure(15);
clf;
for modelIdx = 1:3
    for xx = 1:2
        idx = (xx-1)*3 + modelIdx;
        subplot(2,3,idx);
        switch xx
            case 1            
                imagesc(convertTensorToPlotMatrix(staticEdgeActivation(:,:,:,modelIdx)));
                title([modelNames{modelIdx} ' SE Act'])
            case 2
                imagesc(convertTensorToPlotMatrix(staticEdgeSelectivity(:,:,:,modelIdx)));
                title([modelNames{modelIdx} ' SE Sel']);
        end
        ylabel('Input Noise');
        xlabel('Component Output Noise');
        xticks(1.5:2:num_noises*2);
        xticklabels(noise_range);
        yticks(1.5:2:num_noises*2);
        yticklabels(noise_range);
        colorbar
    end
end

% figure(16);
% clf;
% for modelIdx = 1:3
%     subplot(1,3,modelIdx);
%     imagesc(convertTensorToPlotMatrix(orientationIndex(:,:,:,modelIdx)));
%     title([modelNames{modelIdx}])
%     ylabel('Input Noise');
%     xlabel('Component Output Noise');
%     xticks(1.5:2:num_noises*2);
%     xticklabels(noise_range);
%     yticks(1.5:2:num_noises*2);
%     yticklabels(noise_range);
%     colorbar
% end
% sgtitle('Orientation Index');

outRelPath = [mpath '/../../intermediate_mat_files/'];
listing = dir(outRelPath);
outpath = [listing(1).folder '/'];
save([outpath 'varExpl.mat'],'varExpl','varExplNoisy');
save([outpath 'esidsi_hm.mat'],'esi','dsi');
save([outpath 'staticEdges_hm.mat'],'staticEdgeActivation');
save([outpath 'sparsity_hm.mat'],'sparsity');
save([outpath 'opponency_hm.mat'],'opponencyIndex');
