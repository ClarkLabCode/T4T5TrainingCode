function [models_out, model_index] = GetModel(models, r2_style, varargin)
    var_names = cell(0,1);
    var_values = cell(0,1);
    for ii = 1:2:length(varargin)
        var_names{(ii+1)/2,1} = varargin{ii};
        var_values{(ii+1)/2,1} = varargin{ii+1};
    end
    
    models_out = cell(0,1);
    index = zeros(0,1);
    model_ind = 1;
    
    for pp = 1:length(models)
        keep = true;
        
        for vv = 1:length(var_names)
            if ischar(models{pp}.(var_names{vv}))
                if ~strcmp(models{pp}.(var_names{vv}),var_values{vv})
                    keep = false;
                end
            else
                if models{pp}.(var_names{vv}) ~= var_values{vv}
                    keep = false;
                end
            end
        end
        
        if keep
            models_out{model_ind,1} = models{pp};
            index(model_ind,1) = pp;
            model_ind = model_ind + 1;
        end
    end
    
    if isempty(models_out)
        error('no models have all the requested parameters');
    end

    models_out = [models_out{:}];
    if isfield(models_out,'r2')
        r2tts = cat(1,models_out.r2)';
    else
        r2tts = cat(1,models_out.accuracy)';
    end
    [~,sortInds] = sort(r2tts(end,:));
    models_out = models_out(sortInds);
    model_index = index(sortInds);
    switch r2_style
        case 'all'
            
        case 'max'
            models_out = models_out(end);
            model_index = model_index(end);
    end
end