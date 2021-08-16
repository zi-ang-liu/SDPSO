function [z] = mydist(w,p)

    [S,R] = size(w);
    Q = size(p,2);

    z = zeros(S,Q);

    wt = w';
    for i=1:S
        z(i,:) = sum(bsxfun(@minus,wt(:,i),p).^2,1);
    end

    z = sqrt(z);
end
