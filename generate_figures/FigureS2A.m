mpath = mfilename('fullpath');

params = {};
matfiles = dir([mpath '/../../training_outputs/LN_nonsym/*.mat']);
for xx = 1:length(matfiles)
    thisParams = load(fullfile(matfiles(xx).folder,matfiles(xx).name));
    params = cat(2,params,thisParams.param_dict);
end


chosen_models = GetModel(params,'all');

val_r2s = [];
for ii = 1:length(chosen_models)
    val_r2s = cat(2,val_r2s,chosen_models(ii).val_r2');
    balances(ii) = abs(sum(sign(chosen_models(ii).weights{end})));
end

figure(1);
clf;
hold on;
colors = {'r','g','b'};
%Plot just the first of each type to get the legend to work correctly
for balIdx = 1:3
    thisValR2s = val_r2s(1:100,balances == 2*(balIdx-1));
    plot(thisValR2s(:,1),colors{balIdx});
end
%Plot the rest of the traces
for balIdx = 1:3
    plot(val_r2s(1:100,balances == 2*(balIdx-1)),colors{balIdx});
end
hold off;
ylabel('validation r2');
xlabel('Training Iteration');
legend({'two pairs','3 and 1','4 same'});
ylim([0,0.05]);

chosen_model = chosen_models(end-1);
chosen_model.model_name = 'ln_model';

t_sample_rate = double(chosen_model.sample_freq);
x_step = double(chosen_model.phase_step);

[h, b1, m1, m2, b2, model_structure] = assignModelParams(chosen_model);
figure(2);
clf;
for pp = 1:4
    scaledH(:,:,pp) = h(:,:,pp).*m1(:,pp)';
end

scaledH = fliplr(scaledH);
flippedM1 = flipud(m1);


scaledH = scaledH ./ max(abs(scaledH),[],1);

c_max = max(abs(scaledH(:)));

for pp = 1:4
    %We're plotting input noise as we go up and down, weight noise
    %horizontally
    subplot(1,4,pp);
    pause(0.1);
    t_label = linspace(0,(size(h,1)-1)/t_sample_rate*1000,4);
    x = linspace(0,(size(h,2)-1)*x_step,size(h,2));
    imagesc(x, t_label, scaledH(:,:,pp));
    caxis([-c_max c_max]);
end