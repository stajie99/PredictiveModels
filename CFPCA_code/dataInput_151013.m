%DATA INPUT reads the data in and saves them to a list
%
%   Returns
%   Variables\inputs.mat

% define parameters
clear all;
SPLINE_ORDER                = 4; 
PRESICION_FG_ALG            = 0.0001;
NUMBER_PRINCIPAL_COMPONENTS = 3;
FONT_SIZE                   = 7;

COUNTRY_NAMES               = ['US'; 'UK'; 'EU'; 'JP'];
MATURITIES                  = [1/12,2/12,0.25,4/12,5/12,0.5,...
                               1,2,3,4,5,6,7,8,9,10,12,15,20,25,30];

% create knot sequence
KNOT_SEQUENCE = aptknt(MATURITIES,SPLINE_ORDER);



% initialize data cells
data  = cell(4,1);
dates = cell(4,1);

% import the data for US
[data{1},datesUS] = xlsread('Data\Bloomberg.xlsx','us_ois','A6:V721');
dates{1}          = char(datesUS);
clear datesUS;

% import the data for UK
[data{2},datesUK] = xlsread('Data\Bloomberg.xlsx','sonia_ois','A6:V721');
dates{2}          = char(datesUK);
clear datesUK;
 
% import the data for EU
[data{3},datesEU] = xlsread('Data\Bloomberg.xlsx','eonia_ois','A6:V721');
dates{3}          = char(datesEU);
clear datesEU;

% import the data for Japan
[data{4},datesJapan] = xlsread('Data\Bloomberg.xlsx','tonar_ois','A6:V721');
dates{4}             = char(datesJapan);
clear datesJapan;



%created mean corrected data samples
dataMeanCorrected        = cell(length(data),1);
dateVector               = cell(length(data),1);
for i=1:length(data)
// Subtract the column mean from the corresponding column elements of a matrix A
    dataMeanCorrected{i} = bsxfun(@minus,data{i},mean(data{i}));
    // Convert date and time to vector of components
    dateVector{i}        = datevec(dates{i},'dd/mm/yyyy');
end;

clearvars i dates
save('Variables\inputs.mat');