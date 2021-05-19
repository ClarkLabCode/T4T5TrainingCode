function [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2)
    h = cat(3, h, fliplr(h));
    
    b1 = cat(2, b1, flipud(b1));
    m1 = cat(2, m1, flipud(m1));
    
    b2 = cat(1, b2, b2);
    m2 = cat(1, m2, -m2);
end