function [model_output, component_output] = lnln_model(data_set,h,b1,m1,b2,m2)
    % calculates the output of the LNLN model and all its
    % intermediate steps.

    data_set_gpu = gpuArray(data_set);
    h_gpu = gpuArray(h);
    
    component_output = cell(size(h,3),1);
    channels = zeros([size(data_set,1)-size(h,1)+1,size(data_set,2)-2,size(data_set,3),3],'gpuArray');

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set

            
        for ii = 1:3
            channels(:,:,:,ii) = convn(data_set_gpu(:,ii:end-3+ii,:), h_gpu(:,ii,h_ind), 'valid') + b1(ii, h_ind);
        end

        % rectify the channels
        channels(channels<0) = 0;

        component_output{h_ind} = m1(1,h_ind)*channels(:,:,:,1) + m1(2,h_ind)*channels(:,:,:,2) + m1(3,h_ind)*channels(:,:,:,3);

        
        component_output{h_ind} = component_output{h_ind} + b2(h_ind);
        
        % rectify
        component_output{h_ind}(component_output{h_ind}<0) = 0;
        
        % multiply before summing
        component_output{h_ind} = gather(m2(h_ind)*component_output{h_ind});
    end
    
    model_output = component_output{1};
    
    for cc = 2:length(component_output)
        model_output = model_output + component_output{cc};
    end
    
end

%     component_output = cell(size(h,3),1);
%     channels = zeros(size(data_set,1)-size(h,1)+1,size(data_set,2)-2,3);
% 
%     for h_ind = 1:size(h,3)
%         % convolve each filter in the model with each image in the dev set
%         for ff = 1:size(data_set,3)
%             
%             for ii = 1:3
%                 channels(:,:,ii) = conv2(data_set(:,ii:end-3+ii,ff), h(:,ii,h_ind), 'valid') + b1(ii, h_ind);
%             end
%             
%             % rectify the channels
%             channels(channels<0) = 0;
%             
%             component_output{h_ind}(:,:,ff) = m1(1,h_ind)*channels(:,:,1) + m1(2,h_ind)*channels(:,:,2) + m1(3,h_ind)*channels(:,:,3);
%         end
%         
%         component_output{h_ind} = component_output{h_ind} + b2(h_ind);
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