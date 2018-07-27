

function [nnf,nnfX,nnfY,nnTime] = calcDeepFeatures(Iorig,Torig,net,gpuN)

    I = im2uint8(Iorig);
    T = im2uint8(Torig);
    I = deepFeatures(net,I,gpuN);
    T = deepFeatures(net,T,gpuN);


    sT = size(T);
    sI = size(I);
    Ivec = reshape(I, sI(1)*sI(2), sI(3));
    Tvec = reshape(T, sT(1)*sT(2), sT(3));

    [Tvec, Ivec] = whitening(Tvec, Ivec);

    k=1;
    params.algorithm = 'kdtree';
    params.trees = 8;
    params.checks = 64;
    tic;
    [nnf, distP] = flann_search(Tvec', Ivec',k,params);
    nnTime = toc;

    nnf = reshape(nnf, sI(1:2));

    [nnfY, nnfX] = ind2sub(sI(1:2),nnf);
end


function F_out = normelizeRows(F_in, mean_in, std_in)
    F_out = bsxfun(@minus, F_in, mean_in) ;

    if ~exist('std_in','var')
        F_out = bsxfun(@rdivide, F_out, std_in) ;
    end
end

function [Tvec, Ivec] = whitening(Tvec, Ivec)
    M = mean(Tvec);
    S = std(Tvec);
    S(S<0.001)=1;

    Ivec = normelizeRows(Ivec, M, S);
    Tvec = normelizeRows(Tvec, M, S);
end

