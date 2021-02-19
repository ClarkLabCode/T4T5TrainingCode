%% Find the top performers in each category

mpath = fileparts(mfilename('fullpath'));
chosen_model = load(fullfile(mpath,'..','outputs','output0.mat')).param_dict{1};
chosen_model.model_name = chosen_model.model_function_name;

[h, b1, m1, m2, b2, model_structure] = assignModelParams(chosen_model);

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

%% Plot Filter
fh = figure(1);
pause(0.1);
clf;

t = (0:size(h,1)-1)/t_sample_rate*1000;
t_label = linspace(0,(size(h,1)-1)/t_sample_rate*1000,4);
x = linspace(0,(size(h,2)-1)*x_step,size(h,2));

for pp = 1:num_filt/2
    scaledH(:,:,pp) = h(:,:,pp).*m1(:,pp)';
end

if strcmp(chosen_model.model_name,'ln_model_flip') %ln model use proper convolution in space, other models use xcorr
    scaledH = fliplr(scaledH);
    flippedM1 = flipud(m1);
else
    flippedM1 = m1;
end

c_max = max(abs(scaledH(:)));

for pp = 1:num_filt/2
    subplot(1,2,pp);
    pause(0.1);
    imagesc(x, t_label, scaledH(:,:,pp));
    caxis([-c_max c_max]);
end


%% Responses to edges
pause(0.1);
fh = figure(2);
pause(0.1);
clf;

edgesDataset = GenerateEdgeDataset(1);
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

edgeStrs = {'right light','right dark','left light','left dark'};
for edgeIdx = 1:4
    for compIdx = 1:numComponents/2
        subplot(numComponents/2,4,(compIdx-1)*4 + edgeIdx);
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
