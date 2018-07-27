

addpath(genpath('utils'));
addpath('DIWU_bin');

im1 = imread('boxing1.jpg');
im2 = imread('boxing2.jpg');

Trect = [522 260 62 281];
rectGT = [684 259 54 280];
T = im1(Trect(2):(Trect(2)+Trect(4)),Trect(1):(Trect(1)+Trect(3)),:);

%% run DIWU on RGB features
[scoreMapOut,rectOut,time,nnTime] = runDIWU(0,im2,T,3,0);
showResults(im1, im2, scoreMapOut, Trect, rectOut, rectGT, time, nnTime);


%% run DIWU on deep VGG features
addpath(genpath('matconvnet-1.0-beta24'));
warning('off','MATLAB:colon:nonIntegerIndex');
warning('off','MATLAB:dispatcher:nameConflict');
if ~exist('net','var')
    [ net, gpuN ] = loadNet();
end

[scoreMapOut,rectOut,time,nnTime] = runDIWU(0,im2,T,3,1);
showResults(im1, im2, scoreMapOut, Trect, rectOut, rectGT, time, nnTime);




