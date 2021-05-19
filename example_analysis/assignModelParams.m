function [h, b1, m1, m2, b2, model_structure] = assignModelParams(chosen_model)
    %% extract the weights from the runs
    % get the parameters out based on what model was run
    switch chosen_model.model_name
        case 'ln_model'
            % h are filters
            % b are offsets
            % m are scalar weights
            h = squeeze(chosen_model.weights{1});
            b1 = zeros(size(h,2), size(h,3));
            m1 = ones(size(h,2), size(h,3));
            
            m2 = squeeze(mean(chosen_model.weights{end},2));
            b2 = chosen_model.biases{1}(:)';

            h = rot90(h,2);

            model_structure = @ln_model;

        case 'ln_model_flip'
            h = squeeze(chosen_model.weights{1});
            b1 = zeros(size(h,2), size(h,3));
            m1 = ones(size(h,2), size(h,3));
            
            m2 = squeeze(mean(chosen_model.weights{end},2));
            b2 = chosen_model.biases{1}(:);

            h = rot90(h,2);

            [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2);

            model_structure = @ln_model;

        case 'conductance_model'
            for ff = 1:chosen_model.num_filt
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            for bb = 1:3
                b1(bb, :) = chosen_model.biases{bb};
            end
            
            m2 = squeeze(mean(chosen_model.weights{end},2));

            model_structure = @conductance_model;
            
            m1 = ones(size(h,2), size(h,3));
            b2 = zeros(size(h,3),1);

        case 'conductance_model_flip'
            for ff = 1:size(chosen_model.weights{1},4)
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            if ~isfield(chosen_model,'fit_reversal') || chosen_model.fit_reversal
                m1 = repmat(chosen_model.weights{4}(:), [1, size(h,3)]);

                b2 = chosen_model.weights{5}(:);
            else
                m1 = [-30 60 -30];
                
                b2(4, :) = chosen_model.weights{4}(:);
            end
            
            for bb = 1:3
                b1(bb, :) = chosen_model.biases{bb}';
            end
                
            m2 = squeeze(mean(chosen_model.weights{end},2));
                
            [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2);

            model_structure = @conductance_model;

        case 'lnln_model'
            for ff = 1:size(chosen_model.weights{1},4)
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            m1 = repmat(chosen_model.weights{4}(:), [1, size(h,3)]);
            
            for bb = 1:3
                b1(bb, :) = chosen_model.biases{bb}';
            end
            
            b2 = chosen_model.weights{5}(:);

            m2 = squeeze(mean(chosen_model.weights{end},2));
                
%             [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2);

            model_structure = @lnln_model;
            
        case 'lnln_model_flip'
            for ff = 1:size(chosen_model.weights{1},4)
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            m1 = repmat(chosen_model.weights{4}(:), [1, size(h,3)]);
            
            for bb = 1:3
                b1(bb, :) = chosen_model.biases{bb}';
            end
            
            b2 = chosen_model.weights{5}(:);

            m2 = squeeze(mean(chosen_model.weights{end},2));
                
            [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2);

            model_structure = @lnln_model;

        otherwise
            error('counldnt detect model type');

    end
end
