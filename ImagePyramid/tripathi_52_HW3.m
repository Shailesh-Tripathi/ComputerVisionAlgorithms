clear all;
%% Q1
% Read the input image and convert to grayscale
rgbImage = imread('image.jpg');
grayImage = double(rgb2gray(rgbImage));

% Number of levels in the pyramid
levelCount = 4;
% Calculate appropriate size using Burt & Adelson formula
% R = Mr * 2^N + 1
% C = Mc * 2^N + 1
% If N >= levelCount, the pyramid can be generated. For simplicity, let
% N = leveCount
rows = floor((size(grayImage,1) - 1) / (2^levelCount)) * (2^levelCount) + 1;
cols = floor((size(grayImage,2) - 1) / (2^levelCount)) * (2^levelCount) + 1;
truncatedImage = grayImage(1:rows, 1:cols);

% Display Gaussian and Laplacian Pyramid
GenerateGaussianAndLaplacian(truncatedImage, levelCount);

%% Q2
backgroundImage = double(imread('bg000.bmp'));
objectImage = double(imread('walk.bmp'));

% set the threshold
T =  50;
% Compute the difference and compare with threshold
diffImage = abs(objectImage - backgroundImage) > T;

imshow(diffImage);
pause;

%% Q3
numBg = 30;
% Read all images from file
for i = 1:numBg
    filename = strcat('bg', num2str(i-1, '%03d'), '.bmp');
    backgroundImageArr(:,:,i) = double(imread(filename));
end

% Compute pixel-wise mean of all background images.
meanImage = mean(backgroundImageArr,3);

% Compute standard deviation
sigmaImage = std(backgroundImageArr, 1, 3);%zeros(size(objectImage));

% Compute statistical distance of object pixels from background image.
T = 15;
bsIm = (abs(objectImage - meanImage) ./ sigmaImage) > T;
imshow(bsIm);
pause;

%% Q4
% Dilate the image
d_bsIm = bwmorph(bsIm, 'dilate');
imshow(d_bsIm);
pause;

%% Q5
% Label the boxes using connected components algorithm.
[L, num] = bwlabel(d_bsIm, 8);

% Find the label with largest area using regionprops tool
stat= regionprops(L, 'Area');
[maxArea, maxLabel] =max([stat.Area]);

% Display the largest block only.
maxLabelImage = (L==maxLabel);
imshow(maxLabelImage);

%% Q2- helper functions
% Function to generate the pyramids
function [] = GenerateGaussianAndLaplacian(truncatedImage, levelCount)
    
    a = 0.4;
    window1D = [0.25 - 0.5 * a, 0.25, a, 0.25, .25 - 0.5 * a];
    
    GaussianImageList = cell(1, levelCount);
    LaplacianImageList = cell(1, levelCount);
    
    % Level-1 of Gaussian pyramid is the original image.
    GaussianImageList{1} = truncatedImage;
    
    for level = 2: levelCount

        % Compute corresponding level images of the Gaussian pyramid
        % Level-i Gaussian
        GaussianImageList{level} = BlurAndSample(GaussianImageList{level-1}, window1D);
        
        % Now that the Gaussian image for this level is available, compute
        % Laplacian of previous level.
        LaplacianImageList{level - 1} = GaussianImageList{level-1} - GetInterpolatedImage(GaussianImageList{level});
    end
    
    % The last level of Laplacian pyramid is same as last level of 
    % Gaussian pyramid
    LaplacianImageList{levelCount}  = GaussianImageList{level};
    
    % Display each level of both the pyramids.
    for level = 1: levelCount
        imshowpair(GaussianImageList{level}/255, LaplacianImageList{level}/255, 'montage');
        title(strcat('Gaussian-Laplacian pyramid. Level-',int2str(level)))
        pause;
    end
end

% Function to perform Gaussian blur and sample the image.
function [sampledImage] = BlurAndSample(img, window1D)
    % Gaussian blur
    % Gaussian filter applied along X axis.
    gXIm = imfilter(img, window1D, 'replicate');
    
    % Gaussian filter applied along Y axis.
    gIm = imfilter(gXIm, window1D', 'replicate');
    
    % Sample the image by selecting every 1 out of 2 pixels.
    sampledImage = gIm(1:2:end, 1:2:end);
    
end

% Interpolation function
function [interpolatedImage] = GetInterpolatedImage(image)
    [rows, cols] = size(image);
    interpolatedImage = zeros(2*rows-1, 2*cols-1);
    
    % Copy original values to appropriate indices
    interpolatedImage(1:2:end, 1:2:end) = image(:, :);
    
    % Interpolate along the columns
    interpolatedImage(1:2:end, 2:2:end) = 1/2*(image(:, 1:end-1) + image(:, 2:end));
    
    % Interpolate along the rows.
    % The interpolated values from the previous
    % step is used to further interpolate the middle pixel value(Eg. center
    % pixel in a 5x5 matrix, where only the 4 corners are available. First,
    % we complete the top and bottom row. Then for the use these two rows
    % to interpolate the middle row.
    interpolatedImage(2:2:end, :) = 1/2*(interpolatedImage(1:2:end-2, :) + interpolatedImage(3:2:end, :));
end
