mpath = mfilename('fullpath');

xtPlotFolderRel = [mpath '/../../data_sets'];
listing = dir(xtPlotFolderRel);
xtPlotFolder = [listing(1).folder '/'];
if ~exist('images','var')
    testScenePaths = {[xtPlotFolder 'xtPlot_ns241_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-25_no0_csInf_sy0.mat'],...
                      [xtPlotFolder 'xtPlot_sineWaves_sl20_ll90_pe360_ps5_sf100_tt1_nt6080_hl0-2_vs100_df0-05_no0_csInf.mat'],...
                      };
    images = load(testScenePaths{1});
end

% switch dev to test set if you want
data_set = images.dev_in;
data_ans = images.dev_out;

% set up data_set
data_set = permute(data_set, [2 1 3]);

data_set = (data_set - mean(data_set,[1 2]))./std(data_set,[],[1 2]);

% Format data_ans correctly
data_ans = repmat(permute(data_ans, [1 3 2]), [1 70 1]);
data_ans = data_ans(end-72+1:end, :, :);

%%
matfiles = dir([mpath '/../../training_outputs/NoiseAndModelSweep/*.mat']);
params = {};
for xx = 1:length(matfiles)
    thisParams = load(fullfile(matfiles(xx).folder,matfiles(xx).name));
    params = cat(2,params,thisParams.param_dict);
end

%%

trainedNoises = [0.125, 1];
evalNoises = [0,0.125,0.25,0.5,1];
titles = {'Trained at 0.125','Trained at 1'};

for modelIdx = 1:2
    
    trainedNoise = trainedNoises(modelIdx);

    chosen_models = GetModel(params,'all','model_name','ln_model_flip','output_noise_std',trainedNoise,'input_noise_std',trainedNoise);
    chosen_model = chosen_models(end);
    chosen_model.model_name = chosen_model.model_function_name;

    [h, b1, m1, m2, b2, model_structure] = assignModelParams(chosen_model);

    % get model output
    for inputNoiseIdx = 1:length(evalNoises)
        for outputNoiseIdx = 1:length(evalNoises)
            
            inputNoise = evalNoises(inputNoiseIdx);
            outputNoise = evalNoises(outputNoiseIdx);
            
            data_set_noised = data_set + inputNoise*randn(size(data_set));
            [~, component_output] = model_structure(data_set_noised,h,b1,m1,b2,m2);

            for ii = 1:length(component_output)
                component_output{ii} = component_output{ii} .* createMultNoise(outputNoise,size(component_output{ii}));
            end

            model_output = component_output{1};
            for ii = 2:length(component_output)
                model_output = model_output + component_output{ii};
            end
            
            num = sum((data_ans(:)-model_output(:)).^2);
            denom = sum((data_ans(:)-mean(data_ans(:))).^2);
            varExpl(inputNoiseIdx,outputNoiseIdx,modelIdx) = 1-mean(num./denom);
            
        end
    end
end

figure(1);
clf;
for modelIdx = 1:2

    subplot(1,2,modelIdx);
    varExplCapped = varExpl(:,:,modelIdx);
    varExplCapped(varExplCapped<0) = 0;
    maxVal = max(varExpl(:,:,modelIdx),[],'all');
    imagesc(varExplCapped,[0 maxVal]);
    title(titles{modelIdx});
    xlabel('input noise');
    ylabel('output noise');
    axis image;
    colorbar;
end

function mult_noise = createMultNoise(target_std,noise_size)

    target_mean = 1.0;
    target_var = target_std^2;
    source_mean = log((target_mean^2)/sqrt(target_var + target_mean^2));
    source_std = sqrt(log(target_var/(target_mean^2) + 1));

    mult_noise = exp(source_std*randn(noise_size)+source_mean);

end