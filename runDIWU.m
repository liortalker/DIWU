%------------------------------------------------------------------------%
% Copyright 2018 Lior Talker, Yael Moses and Ilan Shimshoni
% From the paper: "Efficient Sliding Window Computation for
%                  NN-Based Template Matching", ECCV 2018.
% INPUT:
% scoreMode - 0 for DIWU, 1 for IWU
% I - RGB input image
% T - RGB input template
% patchSize - the size of patch for the NN computation
% isDeep - 1 to use vgg deep features, 0 to use RGB features
%
% OUTPUT:
% scoreMapOut - output heatmap
% rectOut - the output bounding box ([top left X, top left Y, width, height])
% time - computation time (not including the NN computation)
% nnTime - NN computation time
%------------------------------------------------------------------------%
function [scoreMapOut,rectOut,time,nnTime] = runDIWU(scoreMode,I,T,patchSize,isDeep,net,gpuN)

    I = im2double(I);
    T = im2double(T);

    sT = size(T);
    %% compute NN field
    nnfApprox = zeros(size(I,1),size(I,2));
    % aproximated params for TreeCANN
    S_grid = 1;
    T_grid = 1;
    S_win = 3;    %must be odd
    T_win = 5;    %must be odd
    eps = 2;
    num_PCA_dims = 9;
    train_patches = 100;
    knn = 5;
    second_phase = 1;

    I = im2uint8(I);
    T = im2uint8(T);
    
    if (isDeep) % VGG features
        [nnf,nnfX,nnfY,nnTime] = calcDeepFeatures(I,T,net,gpuN);
    else % RGB features
        tic;
        [nnf_dist_temp,nnf_X,nnf_Y] = run_TreeCANN(I,T,patchSize,S_grid,T_grid,train_patches,num_PCA_dims,eps,knn,S_win,T_win,second_phase);
        nnTime = toc;
        nnf_X1 = nnf_X(1:end-patchSize+1,1:end-patchSize+1);
        nnf_Y1 = nnf_Y(1:end-patchSize+1,1:end-patchSize+1);
        %remove patchSize from end
        nnfApprox(1:end-patchSize+1,1:end-patchSize+1) = sub2ind(sT(1:2),nnf_Y1,nnf_X1);
        nnf=nnfApprox;
    end

    %% DIWU/IWU computation
    [scoreMapOut, rectOut, time] = computeDIWU(scoreMode,nnf,sT);


end

