function [model_output, component_output] = ln_model(data_set,h,b1,m1,b2,m2)

    data_set_gpu = gpuArray(data_set);
    h_gpu = gpuArray(h);
    b2_gpu = gpuArray(b2);
    m2_gpu = gpuArray(m2);
    
    component_output = cell(size(h_gpu,3),1);

    for h_ind = 1:size(h_gpu,3)
        % convolve each filter in the model with each image in the dev set
        
        component_output{h_ind} = convn(data_set_gpu, h_gpu(:,:,h_ind), 'valid') + b2_gpu(h_ind);

        % rectify
        component_output{h_ind}(component_output{h_ind}<0) = 0;

        % multiply before summing
        component_output{h_ind} = gather(m2_gpu(h_ind)*component_output{h_ind});
    end

    model_output = component_output{1};

    for cc = 2:length(component_output)
        model_output = model_output + component_output{cc};
    end

end

%     tic;
%     component_output = cell(size(h,3),1);
% 
%     for h_ind = 1:size(h,3)
%         % convolve each filter in the model with each image in the dev set
%         for ff = 1:size(data_set,3)
%             component_output{h_ind}(:,:,ff) = conv2(data_set(:,:,ff), h(:,:,h_ind), 'valid') + b2(h_ind);
%         end
%         
%         % rectify
%         component_output{h_ind}(component_output{h_ind}<0) = 0;
%         
%         % multiply before summing
%         component_output{h_ind} = m2(h_ind)*component_output{h_ind};
%     end
%     
%     model_output = component_output{1};
%     
%     for cc = 2:length(component_output)
%         model_output = model_output + component_output{cc};
%     end
%     cpu_time = toc