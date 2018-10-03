%% Q1
sigma=5.0;
faceIm=double(imread('keanu.jpg'));

% MATLAB does not recommends 'imfilter' for gaussian filter as
% 'imgaussfilt' is more optimized than the other. The whole operation can
% be performed in one step instead of two.
%G = fspecial('gaussian', 2*ceil(3*sigma)+1, sigma);
%gIm = imfilter(faceIm, G, 'replicate'); 

gIm = imgaussfilt(faceIm,sigma,'FilterSize', 2*ceil(3*sigma)+1, 'Padding', 'replicate');
% NOTE: Using imgaussfilt instead of fspecial gives very close yet unequal
% results . Relative error = 1.7053e-15. This much error is acceptable.

% Display the results
imshow(gIm/255);  % double images need range of 0-1

imwrite(uint8(gIm), 'gIm.bmp');

pause;

%% Q2

% Use gaussDeriv2D to get the Gaussian Derivative Mask
[Gx, Gy] = gaussDeriv2D(sigma); % Defined at the end

% Plot the surface graph
surf(Gx);
colormap(jet);
xlabel('x')
ylabel('y')
title('Gx');
pause;

% Plot 2D image
imagesc(Gx); colormap('gray');
pause;

% Plot the surface graph
surf(Gy);
colormap(jet);
xlabel('x')
ylabel('y')
title('Gy');
pause;

% Plot 2D image
imagesc(Gy); colormap('gray');
pause;

%% Q3
% Gaussian derivative is used in autonomous vehicles to detect other
% vehicles. Here is an example.
rgbIm=(imread('car.jpg'));

% Use gaussian derivative with smaller sigma value to get better results
sigma = 1.0;
[Gx, Gy] = gaussDeriv2D(sigma);

% Convert RGB image to grayscale
Im = double(rgb2gray(rgbIm));

% Apply Gx filter
gxIm = imfilter(Im, Gx, 'replicate');
% Apply Gy filter
gyIm = imfilter(Im, Gy, 'replicate');

% Compute magnitude to get the resultant image.
magIm = sqrt(gxIm.^2 + gyIm.^2);

% Display the image. Double images need range 0-1
imagesc(magIm/255);
axis('image');
colormap('gray');

pause;

%% Q4 
% Set threshold level
T = 30;

% Convert grayscale image to bianry using a threshold value.
tIm = magIm > T;

imagesc(tIm);
axis('image');
colormap('gray');
pause;

%% Q5 
% Apply sobel mask. An ideal mask should be normalized to 1. Hence divided
% by 8
Fx = -fspecial('sobel')/8;
fxIm = imfilter(Im,Fx);
Fy=-fspecial('sobel');
fyIm = imfilter(Im,Fy)/8;

% Calculate magnitude
magIm = sqrt(fxIm.^2 + fyIm.^2);

% Different threshold for sobel mask
T = 50;
tIm = magIm > T;
imagesc(tIm);
axis('image');
colormap('gray');

pause;

%% Q6
% Canny Edge. Using the in-built function provided by matlab
tIm = edge(Im, 'canny');
imagesc(tIm);
axis('image');
colormap('gray');

%% Q2 function
function [Gx, Gy] = gaussDeriv2D(sigma)
    % Calculate appropriate length of the square mask.
    % Mask will have length (3*sigma) points on both sides of origin
    length = ceil(3*sigma);
    
    % Initialize size of mask
    Gx = zeros(2*length +1, 2*length +1);
    Gy = zeros(2*length +1, 2*length +1);
    
    for x = -length : length
        for y = - length : length     
            % Assume x0 = 0 and y0 = 0 (both mean x and mean y are zero)
            % Calculate the component common to both Gx and Gy
            e = exp(-(x*x + y*y)/(2*sigma*sigma)) ;
            e = e / (2 * pi * (sigma^4));

            % Multiply x and e for Gx
            % Note that x-coordinate would be the 2nd dimension in MATLAB
            % and y-coordinate would the first coordinate.
            Gx(y + length +1, x + length +1) = x * e;

            % Multiply y and e for Gy
            Gy(y + length +1, x + length +1) = y * e;
        end
    end
end

