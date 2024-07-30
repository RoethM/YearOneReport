clear all

% Importing spreadsheet and selecting required columns 
filename = "C:\Users\ppxmr2\OneDrive - The University of Nottingham\Documents\Coding projects\MOLLI\Spreadsheets\Pancreas_data\Pancreas_csv_specifics.csv"; %Set to correct file name
SheetData = readtable(filename);

T1InputMedullaL = table2array(SheetData(:, "T1MedullaL")); %Set to correct column for L medulla T1 values
T1InputMedullaL(isnan(T1InputMedullaL))=0;

T1InputCortexL = table2array(SheetData(:, "T1CortexL")); 
T1InputCortexL(isnan(T1InputCortexL))=0;

T1InputMedullaR = table2array(SheetData(:, "T1MedullaR")); 
T1InputMedullaR(isnan(T1InputMedullaR))=0;

T1InputCortexR = table2array(SheetData(:, "T1CortexR")); 
T1InputCortexR(isnan(T1InputCortexR))=0;

HRInputKidney = table2array(SheetData(:, "kidney_mean_hr")); %Set to correct column for RR values
HRInputKidney(isnan(HRInputKidney))=0;


T1InputPancreas = table2array(SheetData(:, "T1Pancreas")); 
T1InputPancreas(isnan(T1InputPancreas))=0;

HRInputPancreas = table2array(SheetData(:, "pancreas_mean_hr")); %Set to correct column for RR values
HRInputPancreas(isnan(HRInputPancreas))=0;


T1InputLiver = table2array(SheetData(:, "T1Liver")); 
T1InputLiver(isnan(T1InputLiver))=0;

HRInputLiver = table2array(SheetData(:, "liver_mean_hr")); %Set to correct column for RR values
HRInputLiver(isnan(HRInputLiver))=0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Running kidney values 

% Creating Timestamps for Simulink input
InputLength = length(T1InputMedullaL);
Timegap = 10/InputLength;
TimeStamps = (0:Timegap:10);
TimeStamps = reshape(TimeStamps, [length(TimeStamps), 1]);
TimeStamps = TimeStamps(1:end-1, :);

% Adding timestamps to T1/HR arrays for Simulink input
SimInputT1 = [TimeStamps, T1InputMedullaL];
SimInputHR = [TimeStamps, HRInputKidney];

mdl1 = 'LUT_kidney_25';

% Check if corrected values already exist, skips running sim if they do
ColumnCheck = ismember('T1MedullaLCorrected_25',SheetData.Properties.VariableNames);

if ColumnCheck == 0
    % Running inputs through Simulink
    SimOutKidney = sim(mdl1);
    T1MedullaLCorrected = SimOutKidney.simout;
    T1MedullaLCorrected = T1MedullaLCorrected(1:InputLength, :);
    SheetData = addvars(SheetData, T1MedullaLCorrected, 'After', 'T1MedullaL','NewVariableNames','T1MedullaLCorrected_25');
end



SimInputT1 = [TimeStamps, T1InputCortexL];

% Check if corrected values already exist, skips running sim if they do
ColumnCheck = ismember('T1CortexLCorrected_25',SheetData.Properties.VariableNames);

if ColumnCheck == 0
    % Running inputs through Simulink
    SimOutKidney = sim(mdl1);
    T1CortexLCorrected = SimOutKidney.simout;
    T1CortexLCorrected = T1CortexLCorrected(1:InputLength, :);
    SheetData = addvars(SheetData, T1CortexLCorrected, 'After', 'T1CortexL','NewVariableNames','T1CortexLCorrected_25');
end



SimInputT1 = [TimeStamps, T1InputMedullaR];

% Check if corrected values already exist, skips running sim if they do
ColumnCheck = ismember('T1MedullaRCorrected_25',SheetData.Properties.VariableNames);

if ColumnCheck == 0
    % Running inputs through Simulink
    SimOutKidney = sim(mdl1);
    T1MedullaRCorrected = SimOutKidney.simout;
    T1MedullaRCorrected = T1MedullaRCorrected(1:InputLength, :);
    SheetData = addvars(SheetData, T1MedullaRCorrected, 'After', 'T1MedullaR','NewVariableNames','T1MedullaRCorrected_25');
end



SimInputT1 = [TimeStamps, T1InputCortexR];

% Check if corrected values already exist, skips running sim if they do
ColumnCheck = ismember('T1CortexRCorrected_25',SheetData.Properties.VariableNames);

if ColumnCheck == 0
    % Running inputs through Simulink
    SimOutKidney = sim(mdl1);
    T1CortexRCorrected = SimOutKidney.simout;
    T1CortexRCorrected = T1CortexRCorrected(1:InputLength, :);
    SheetData = addvars(SheetData, T1CortexRCorrected, 'After', 'T1CortexR','NewVariableNames','T1CortexRCorrected_25');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Running pancreas values 

% Adding timestamps to T1/HR arrays for Simulink input
SimInputT1 = [TimeStamps, T1InputPancreas];
SimInputHR = [TimeStamps, HRInputPancreas];

mdl2 = 'LUT_liver_pancreas_25';

% Check if corrected values already exist, skips running sim if they do
ColumnCheck = ismember('T1PancreasCorrected_25',SheetData.Properties.VariableNames);

if ColumnCheck == 0
    % Running inputs through Simulink
    SimOutPancreas = sim(mdl2);
    T1PancreasCorrected = SimOutPancreas.simout;
    T1PancreasCorrected = T1PancreasCorrected(1:InputLength, :);
    SheetData = addvars(SheetData, T1PancreasCorrected, 'After', 'T1Pancreas','NewVariableNames','T1PancreasCorrected_25');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Running liver values 

% Adding timestamps to T1/HR arrays for Simulink input
SimInputT1 = [TimeStamps, T1InputLiver];
SimInputHR = [TimeStamps, HRInputLiver];

% Check if corrected values already exist, skips running sim if they do
ColumnCheck = ismember('T1LiverCorrected_25',SheetData.Properties.VariableNames);

if ColumnCheck == 0
    % Running inputs through Simulink
    SimOutLiver = sim(mdl2);
    T1LiverCorrected = SimOutLiver.simout;
    T1LiverCorrected = T1LiverCorrected(1:InputLength, :); %Removes first value of 0
    SheetData = addvars(SheetData, T1LiverCorrected, 'After', 'T1Liver','NewVariableNames','T1LiverCorrected_25');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Writing results to spreadsheet
writetable(SheetData, filename) %,"AutoFitWidth",false);

