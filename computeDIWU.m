%------------------------------------------------------------------------%
% Copyright 2018 Lior Talker, Yael Moses and Ilan Shimshoni
% From the paper: "Efficient Sliding Window Computation for
%                  NN-Based Template Matching", ECCV 2018.
% INPUT:
% scoreMode - 0 for DIWU, 1 for IWU
% nnf - NN field W X H X 2 (x ind and y ind for each patch)
% sT - size of the template
% h - bandwidth parameter (usually 1)
%
% OUTPUT:
% scoreMapOut - output heatmap
% rectOut - the output bounding box ([top left X, top left Y, width, height])
% time - computation time (not including the NN computation)
%------------------------------------------------------------------------%
function [scoreMapOut, rectOut, time] = computeDIWU(scoreMode, nnf, sT)

    h = 1;
    [ySrc,xSrc] = ind2sub(sT, (0:(sT(1)*sT(2)))');
    xyPositions = [xSrc(:), ySrc(:)]';
    xyPositions = xyPositions -1; % updating to cpp indexing

    %% weighting the NN based on popularity
    tic;
    % compute weights based in the image context
    [nnfCountIm,nnfCountIdxIm] = ismember(nnf,1:(sT(1)*sT(2)));
    [idfRaw,idfTempMat] = histc(nnfCountIdxIm(:),1:(sT(1)*sT(2)));
    idfTempMat2 = 100000*ones(1,length(idfTempMat));
    idfTempMat2(idfTempMat~=0) = idfRaw(idfTempMat(idfTempMat~=0));
    idfMat = reshape(idfTempMat2,size(nnf));

    pixelwiseInvIdfMat = exp(-(1/10)*idfMat);
    if (max(max(pixelwiseInvIdfMat)) == 0)
        pixelwiseInvIdfMat = 1./idfMat;
    end

    newNnf = sT(1)*sT(2)*ones(size(nnf,1),size(nnf,2));
    newNnf(2:(end-1),2:(end-1)) = nnf(2:(end-1),2:(end-1));

    %% calling c++
    if (scoreMode == 0)
        tic;
        scoreMapOut = DIWU('DIWU', int32(newNnf), int32(sT(1)), int32(sT(2)), int32(xyPositions), h, single(pixelwiseInvIdfMat)); %mex version
        time = toc;
    else
        invIdfMat = conv2(pixelwiseInvIdfMat,ones(sT(1),sT(2)),'valid');
        time = toc;
        scoreMapOut = invIdfMat;
    end

    %% find the target
    padMap = padding( ones(size(scoreMapOut)) , sT(1:2) );
    scoreMapOut = padding(scoreMapOut,sT(1:2));

    windowSizeDividor = 3;
    locSearchStyle = '';
    rectOut = findTargetLocation(scoreMapOut,locSearchStyle,[sT(2) sT(1)], windowSizeDividor, true, padMap);

end
