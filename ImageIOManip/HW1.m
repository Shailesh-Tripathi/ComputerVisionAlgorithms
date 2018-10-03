clear all;

% Q1
%% Operations on gray image.
% Read the image
grayIm = imread('buckeyes_gray.bmp');

% Use full range of the colormap. Basically mapping the values from 0-255
imagesc(grayIm);

% Plot with the actual dimension (by setting the same scale for each
% dimension)
axis('image');

% Plot image in grayscale
colormap('gray');

% Write image to file in jpg format.
% Note that this is lossy format. The loss can be avoided using:
% imwrite(grayIm, 'buckeyes_gray.jpg', 'Mode', 'lossless');
imwrite(grayIm, 'buckeyes_gray.jpg');

% Wait for key press
pause;

%% Operations on RGB image.
% Read the image
rgbIm = imread('buckeyes_rgb.bmp');

% Use full range of the colormap. Basically mapping the values from 0-255
imagesc(rgbIm);

% Plot with the actual dimension (by setting the same scale for each
% dimension)
axis('image');

% Write image to file in jpg format.
% Note that this is lossy format. The loss can be avoided using:
% imwrite(grayIm, 'buckeyes_rgb.jpg', 'Mode', 'lossless');
imwrite(rgbIm, 'buckeyes_rgb.jpg');

pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Q2

% Convert RGB image to grayscale
grayIm = rgb2gray(rgbIm);
 
% Display the image after setting the axis and colormap
imagesc(grayIm);
axis('image');
colormap('gray');

pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Q3

% Create 10x10 matrix of zeros
zBlock = zeros(10, 10);

% Create a 10x10 matrix of ones and multiply each value by 255
oBlock = ones(10, 10) * 255;

% Create a 20x20 matrix by stitching black and white alternatively.
pattern = [zBlock oBlock; oBlock zBlock];

% Repeat the pattern in both directions 5 times to create a big checker
% board.
checkerIm = repmat(pattern, 5, 5);

% First convert the double-type matrix to uint8. Since all values will be
% within 0-255, therefore no loss of data.
% Then write the image to file.
imwrite(uint8(checkerIm), 'checkerIm.bmp');

% Read the image back from file and display it.
Im = imread('checkerIm.bmp');
imagesc(Im)
colormap('gray')
axis('image'); 
