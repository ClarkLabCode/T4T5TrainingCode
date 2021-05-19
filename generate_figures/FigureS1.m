%% Find the top performers in each category

mpath = mfilename('fullpath');

for ii = 1:2
    figure(ii);
    clf;
end
%%
outputNames = {'LN_highestnoise','LNLN_highestnoise','conductance_highestnoise'};
runFolders = {'NoiseAndModelSweep','NoiseAndModelSweep','NoiseAndModelSweep'};
noiseLevels = [1,1,1];
modelNames = {'ln_model_flip','lnln_model_flip','conductance_model_flip',};
modelNamesPrinting = {'ln','lnln','synaptic'};
for paramIdx = 1:length(outputNames)
    params = {};
    matfiles = dir([mpath '/../../training_outputs/' runFolders{paramIdx} '/*.mat']);
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
    
    %% Contrast
    figure(1);

    hold on;
    contrasts = linspace(0,1,20);
    sineContrastDataset = GenerateSineDataset(60*ones(size(contrasts)),60,2).*reshape(contrasts,1,1,[])*GetSineScaleFactor(60);
    hUpscaled = interp1(1:30,h,linspace(1,30,300));
    [model_vels_output, component_sine_output] = model_structure(sineContrastDataset,hUpscaled,b1,m1,b2,m2);
    modelAvRespVel = squeeze(mean(model_vels_output,[1 2]));

    plot(contrasts,modelAvRespVel)
    xlabel('Contrast');
    ylabel('Model Average Response (deg/s)');
    
    
    
    %% TF Tuning
    figure(2);

    
    gpuDevice(1);
    reset(gpuDevice(1));
    resolution = 25;
    
    sineVelDataset = [];
    lambdas = [22.5, 45, 90];
    velocities = 15*2.^(linspace(0,6,resolution));
    for ii = 1:length(lambdas)
        sineVelDataset = cat(3,sineVelDataset,GenerateSineDataset(velocities,lambdas(ii),2)*0.5*GetSineScaleFactor(lambdas(ii)));
    end
    hUpscaled = interp1(1:30,h,linspace(1,30,300));
    [model_vels_output, component_sine_output] = model_structure(sineVelDataset,hUpscaled,b1,m1,b2,m2);
    avRespVel = reshape(mean(model_vels_output,[1 2]),length(velocities),length(lambdas));

    subplot(3,2,2 + 2*(paramIdx-1))
    semilogx(velocities,avRespVel);
    xlabel('vel');
    xticks(velocities(1:8:end));
    ylabel([modelNamesPrinting{paramIdx} ' resp']);
    
    
    sineTFDataset = [];
    TFs = 2.^(linspace(-1,5,resolution));
    for ii = 1:length(lambdas)
        sineTFDataset = cat(3,sineTFDataset,GenerateSineDataset(lambdas(ii)*TFs,lambdas(ii),2)*0.5*GetSineScaleFactor(lambdas(ii)));
    end
    hUpscaled = interp1(1:30,h,linspace(1,30,300));
    [model_TFs_output, component_sine_output] = model_structure(sineTFDataset,hUpscaled,b1,m1,b2,m2);
    avRespTF = reshape(mean(model_TFs_output,[1 2]),length(TFs),length(lambdas));

    subplot(3,2,1 + 2*(paramIdx-1));
    semilogx(TFs,avRespTF);
    xlabel('TF');
    xticks(TFs(1:8:end));
    ylabel([modelNamesPrinting{paramIdx} ' resp']);
    
end
figure(1);
legend(modelNamesPrinting);
figure(2);