clear;

velocity = 45;
stimTime = 4.01;
data_set = GenerateSineDataset([velocity,-velocity],45,stimTime,0,0.01);

% Format data_ans correctly
data_ans = repmat(cat(3,velocity,-velocity), [stimTime*100 72 1]);

%%
mpath = mfilename('fullpath');
matfiles = dir([mpath '/../../training_outputs/NoiseAndModelSweep/*.mat']);
params = {};
for xx = 1:length(matfiles)
    thisParams = load(fullfile(matfiles(xx).folder,matfiles(xx).name));
    params = cat(2,params,thisParams.param_dict);
end

%%
figure(1);
clf;
sgtitle('Noise levels:');

injectedNoises = [0,0,1,1];
trainedNoises = [0.125, 1, 0.125,1];
titles = {'0.125 Trained, 0 Eval','1 Trained, 0 Eval','0.125 Trained, 1 Eval','1 Trained, 1 Eval'};

for modelIdx = 1:4
    
    trainedNoise = trainedNoises(modelIdx);
    injectedNoise = injectedNoises(modelIdx);

    chosen_models = GetModel(params,'all','model_name','ln_model_flip','output_noise_std',trainedNoise,'input_noise_std',trainedNoise);
    chosen_model = chosen_models(end);
    chosen_model.model_name = chosen_model.model_function_name;

    [h, b1, m1, m2, b2, model_structure] = assignModelParams(chosen_model);

    
    
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
    
    %%
    % get model output
    chosenTrace = 1;
    data_set_noised = data_set + injectedNoise*randn(size(data_set));
    [~, component_output] = model_structure(data_set_noised,h,b1,m1,b2,m2);

    for ii = 1:length(component_output)
        component_output{ii} = component_output{ii} .* createMultNoise(injectedNoise,size(component_output{ii}));
    end

    model_output = component_output{1};
    for ii = 2:length(component_output)
        model_output = model_output + component_output{ii};
    end

    subplot(3,4,modelIdx);
    tt = linspace(0,stimTime*1000,stimTime*100-29);
    dataSegment = data_set_noised(end-length(tt)+1:end,1:3,chosenTrace);
    plot(tt,dataSegment);
    xlabel('time (ms)')
    ylabel('input contrast');
    if modelIdx == 1
        legend({'1','2','3'});
    end
    title(titles{modelIdx});

    subplot(3,4,4+modelIdx);
    component_output_traces = [];
    for ii = 1:4
        component_output_traces = cat(2,component_output_traces,component_output{ii}(:,1,chosenTrace));
    end
    plot(tt,[velocity*ones(length(tt),1),model_output(:,1,chosenTrace), component_output_traces])
    xlabel('time (ms)');
    ylabel('velocity (deg/s)');
    if modelIdx == 1
%         legend({'true','pred.','A','B'});
    legend({'true','pred.','A+','B+','A-','B-'});
    end

    subplot(3,4,8+modelIdx);
    hold on;
    plot(tt,model_output(:,1,chosenTrace),'Color', [0 0 0]);
    lineHandles = plot(tt,model_output(:,:,chosenTrace),'Color', [0.8 0.8 0.8]);
    for ii = 1:length(lineHandles)
        lineHandles(ii).Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    plot(tt,velocity*ones(length(tt),1),'b');
    plot(tt,mean(model_output(:,:,chosenTrace),2),'r');
    hold off;
    xlabel('time (ms)');
    ylabel('velocity (deg/s)');
    if modelIdx == 1
        legend({'pred.','true','pred. mean'});
    end
end

function [groupTrueMeans, groupPredMeans, groupPredSems] = calcFitLine(true,pred)

    groupEdges = quantile(true(:),99);
    groupEdges = [-inf, groupEdges, inf];
    [~,~,bins] = histcounts(true(:),groupEdges);
    groupTrueMeans = accumarray(bins,true(:),[],@mean);
    groupPredMeans = accumarray(bins,pred(:),[],@mean);
    groupPredSems = accumarray(bins,pred(:),[],@(x) mean(x) / sqrt(length(x)));
    

end

function plotErrorBands(x,y,e)
    lo = y - e;
    hi = y + e;

    hp = patch([x; x(end:-1:1); x(1)], [lo; hi(end:-1:1); lo(1)], 'r');
    hold on;
    hl = line(x,y);

    set(hp, 'facecolor', [1 0.8 0.8], 'edgecolor', 'none');
    set(hl, 'color', 'r', 'marker', 'x');
end

function mult_noise = createMultNoise(target_std,noise_size)

    target_mean = 1.0;
    target_var = target_std^2;
    source_mean = log((target_mean^2)/sqrt(target_var + target_mean^2));
    source_std = sqrt(log(target_var/(target_mean^2) + 1));

    mult_noise = exp(source_std*randn(noise_size)+source_mean);

end