function matrix = convertTensorToPlotMatrix(tensor)
    dimExpand = sqrt(size(tensor,1));
    matrix = zeros(dimExpand*size(tensor,[2 3]));
    for ii = 1:size(tensor,3)
        for jj = 1:size(tensor,2)
            for kk = 1:size(tensor,1)
                row = mod((kk-1),dimExpand) + dimExpand*(jj-1);
                col = floor((kk-1)/dimExpand) + dimExpand*(ii-1);
                matrix(row+1,col+1) = tensor(kk,jj,ii);
            end
        end
    end
end