%% This file is to explain how to reproduce the following csv files:
%{
1.'dataTable_test_2D_124810kdata.csv', needed in Fig 10 and Fig S11.
  'dataTable_optm_2D_124810kdata.csv', needed in Fig 10 and Fig S11.
  'dataTable_1248kdata_forFigureP.csv', needed in Fig8 and Fig9.
2.'dataTable_test_3D_124810kdata.csv', needed in Fig 10 and Fig S10.
  'dataTable_optm_3D_124810kdata.csv', needed in Fig 10 and Fig S10.
3.'dataTable_test_3D_500kdata.csv'  , needed in Fig 10 and Fig S10.
    'dataTable_optm_3D_500kdata.csv', needed in Fig 10 and Fig S10.
    'dataTable_test_2D_500kdata.csv'  , needed in Fig 10 and Fig S10.
    'dataTable_optm_2D_500kdata.csv', needed in Fig 10 and Fig S10.




The above files are already generated and saved in the folder, so for the
reader who only want to reproduce the plotting of the figure, please ignore
ALL of the current file, and just use the matlab function
'plot_every_figure.m'.

%}

%{
These files are used to plot figures for the paper. They are already saved
so it is easy and quick to plot the figures using . But in case readers want t



Because many files are big (for example, each trained network could be as 
large as 220MB, and simulated datasets are even larger),
we cannot upload all of them, so here we just upload all the small files
with which one can reproduce those big files. Of course, the obtained
values could be slightly different when being run on a different device
with different gpu's, but that should not qualitatively change the results.

list of contents:
1.How to make 'dataTable_test_2D_124810kdata.csv',
    and 'dataTable_optm_2D_124810kdata.csv',
    'dataTable_1248kdata_forFigureP.csv'
    search 'makefile01' with Ctrl+F
2. How to make 'dataTable_test_2D_500kdata.csv', search 'makefile02'
3.How to make 'dataTable_optm_2D_500kdata.csv', search 'makefile03'
4.How to make 'dataTable_test_3D_124810kdata.csv' and 
'dataTable_optm_3D_124810kdata.csv' , search 'makefile04'
5. 'dataTable_test_3D_500kdata.csv' and  'dataTable_optm_3D_500kdata.csv',
search 'makefile05'


...
%}







%% 'makefile01'
%{
How to make:
'dataTable_test_2D_124810kdata.csv' and dataTable_optm_2D_124810kdata.csv'
and dataTable_1248kdata_forFigureP.csv
%}


%{
Here is step-by-step guidance on how to get those csv's:

Step 1: Use **** to generate parameter samples
Step 2: Use **** to integrate
Step 3: Use *** to get the training dataset
Step 4: Make sure the 



The file is obtained by running the following codes in this section. (Use
'run section', do not click 'run' which will run the whole file!)
But before running, one should prepare several files.
One need to prepare several ckpt files saved in /model CKPT files/250207/
and /model CKPT files/250205/, and 
'parainfertest_Mvlognormaldirect_250205*.csv' where * is from 001 to 040;
'parainfertest_Mvlognormaldirect_250207*.csv' where * is from 001 to 064;
'paratest_Mvlognormaldirect_250205*.csv' where * is from 001 to 040;
'paratest_Mvlognormaldirect_250207*.csv' where * is from 001 to 064;




%}
for kk=1:1 %%% this trivial for loop makes it able to be collapsed; 
    fileList = dir('model CKPT files/250207/*.ckpt');
    numFiles = length(fileList);
    dataMatrix = cell(numFiles, 6);
    
    for i = 1:numFiles
        filename = fileList(i).name;
        jobNumMatch = regexp(filename, '_(250207\d{3})_', 'tokens');
        if ~isempty(jobNumMatch)
            dataMatrix{i,1} = str2double(jobNumMatch{1}{1});
        else
            dataMatrix{i,1} = NaN; % If no match, assign NaN
        end
    
        if contains(filename, 'FX2DNN')
            dataMatrix{i,2} = '2D';
        elseif contains(filename, 'GFX3DNN')
            dataMatrix{i,2} = '3D';
        else
            dataMatrix{i,2} = 'Unknown';
        end
        
        % Extract npara value
        nparaMatch = regexp(filename, '_npara(\d+)_', 'tokens');
        if ~isempty(nparaMatch)
            dataMatrix{i,3} = str2double(nparaMatch{1}{1});
        else
            dataMatrix{i,3} = NaN;
        end
    
         
        if dataMatrix{i,1}<=250207040 && dataMatrix{i,1}>=250207033
            dataMatrix{i,4} = 1000;
        elseif dataMatrix{i,1}<=250207048 && dataMatrix{i,1}>=250207041
            dataMatrix{i,4} = 2000;
        elseif dataMatrix{i,1}<=250207056 && dataMatrix{i,1}>=250207049
            dataMatrix{i,4} = 4000;
        elseif dataMatrix{i,1}<=250207064 && dataMatrix{i,1}>=250207057
            dataMatrix{i,4} = 8000;
    
        elseif dataMatrix{i,1}>=250207001 && dataMatrix{i,1}<=250207008
            dataMatrix{i,4} = 1000;
        elseif dataMatrix{i,1}>=250207009 && dataMatrix{i,1}<=250207016
            dataMatrix{i,4} = 2000;
        elseif dataMatrix{i,1}>=250207017 && dataMatrix{i,1}<=250207024
            dataMatrix{i,4} = 4000;
        elseif dataMatrix{i,1}>=250207025 && dataMatrix{i,1}<=250207032
            dataMatrix{i,4} = 8000;
    
    
        end
    
        
        % Extract feature type (noFE, catonlytGIF, twithreciGIF, or fullreci)
        if contains(filename, 'noFE')
            dataMatrix{i,5} = 'fta_noFE';
        elseif contains(filename, 'catonlytGIF')
            dataMatrix{i,5} = 'ftb_catonlytGIF';
        elseif contains(filename, 'twithreciGIF')
            dataMatrix{i,5} = 'ftc_twithreciGIF';
        elseif contains(filename, 'fullreci')
            dataMatrix{i,5} = 'ftd_fullreci';
        else
            dataMatrix{i,5} = 'Unknown';
        end
        
        % Extract activation function (relu or tanh01)
        if contains(filename, '_relu_')
            dataMatrix{i,6} = 'relu';
        elseif contains(filename, '_tanh01_')
            dataMatrix{i,6} = 'tanh01';
        else
            dataMatrix{i,6} = 'Unknown';
        end
        
        % Extract val_loss value
    %     valLossMatch = regexp(filename, '-val_loss=([\d\.]+)', 'tokens');
        valLossMatch = regexp(filename, '-val_loss=([0-9]*\.?[0-9]+)', 'tokens');
    
        if ~isempty(valLossMatch)
            dataMatrix{i,7} = str2double(valLossMatch{1}{1});
        else
            dataMatrix{i,7} = NaN;
        end
    end
    
    % Convert cell array to table and display results
    columnNames = {'JobNumber', 'Dimension', 'npara','ntrain', 'FeatureType', 'Activation', 'ValLoss'};
    dataTable_1248kdata = cell2table(dataMatrix, 'VariableNames', columnNames);
    
    % Sort dataTable based on multiple columns in ascending order
    dataTable_1248kdata = sortrows(dataTable_1248kdata, {'JobNumber', 'Dimension', 'npara', 'FeatureType', 'Activation'});
     

    dataTable_3D_1248kdata=dataTable_1248kdata(33 :64,:);
    dataTable_2D_1248kdata=dataTable_1248kdata(1:32,:);
     
    fileList = dir('model CKPT files/250205/*.ckpt');
    numFiles = length(fileList);
    
    % Initialize cell array to store extracted data
    dataMatrix = cell(numFiles, 6);
    
    for i = 1:numFiles
        filename = fileList(i).name;
        
        % Extract 8-digit job number (starting with 250203 or 250131)
        jobNumMatch = regexp(filename, '_(250205\d{3})_', 'tokens');
        if ~isempty(jobNumMatch)
            dataMatrix{i,1} = str2double(jobNumMatch{1}{1});
        else
            dataMatrix{i,1} = NaN; % If no match, assign NaN
        end
        
        % Determine if it is 2D or 3D
        if contains(filename, 'FX2DNN')
            dataMatrix{i,2} = '2D';
        elseif contains(filename, 'GFX3DNN')
            dataMatrix{i,2} = '3D';
        else
            dataMatrix{i,2} = 'Unknown';
        end
        
        % Extract npara value
        nparaMatch = regexp(filename, '_npara(\d+)_', 'tokens');
        if ~isempty(nparaMatch)
            dataMatrix{i,3} = str2double(nparaMatch{1}{1});
        else
            dataMatrix{i,3} = NaN;
        end
    
        dataMatrix{i,4}=10000;
        
        % Extract feature type (noFE, catonlytGIF, twithreciGIF, or fullreci)
        if contains(filename, 'noFE')
            dataMatrix{i,5} = 'fta_noFE';
        elseif contains(filename, 'catonlytGIF')
            dataMatrix{i,5} = 'ftb_catonlytGIF';
        elseif contains(filename, 'twithreciGIF')
            dataMatrix{i,5} = 'ftc_twithreciGIF';
        elseif contains(filename, 'fullreci')
            dataMatrix{i,5} = 'ftd_fullreci';
        else
            dataMatrix{i,5} = 'Unknown';
        end
        
        % Extract activation function (relu or tanh01)
        if contains(filename, '_relu_')
            dataMatrix{i,6} = 'relu';
        elseif contains(filename, '_tanh01_')
            dataMatrix{i,6} = 'tanh01';
        else
            dataMatrix{i,6} = 'Unknown';
        end
        
        % Extract val_loss value
        valLossMatch = regexp(filename, '-val_loss=([0-9]*\.?[0-9]+)', 'tokens');
    
        if ~isempty(valLossMatch)
            dataMatrix{i,7} = str2double(valLossMatch{1}{1});
        else
            dataMatrix{i,7} = NaN;
        end
    end
    
    % Convert cell array to table and display results
    columnNames = {'JobNumber', 'Dimension', 'npara','ntrain', 'FeatureType', 'Activation', 'ValLoss'};
    dataTable_10kdata = cell2table(dataMatrix, 'VariableNames', columnNames);
    
    % Sort dataTable based on multiple columns in ascending order
    dataTable_10kdata = sortrows(dataTable_10kdata, {'JobNumber', 'Dimension', 'npara','ntrain', 'FeatureType', 'Activation'});
     
     
    dataTable_3D_10kdata=dataTable_10kdata(1 :24,:);
    dataTable_2D_10kdata=dataTable_10kdata(25:40,:);
    
     
    dataTable_test_2D_1248kdata=dataTable_2D_1248kdata;
    R2_columnNames ={'Sfal', 'Pxa', 'Sffb', 'PXFCR', 'Fb'};
    numFiles = 32;  % Number of jobs
    R2_values = nan(numFiles, 5);  % 5columns for 5 parameters
    
    for i = 1:numFiles
        jobNumber = dataTable_test_2D_1248kdata.JobNumber(i);
        paratest_file = sprintf('para_and_parainfer/paratest_Mvlognormaldirect_%d.csv', jobNumber);
        parainfertest_file = sprintf('para_and_parainfer/parainfertest_Mvlognormaldirect_%d.csv', jobNumber);
    %     parainferfromoptm_file = sprintf('parainferfromoptm_Mvlognormaldirect_%d.csv', jobNumber);
        if exist(paratest_file, 'file') && exist(parainfertest_file, 'file')% && exist(parainferfromoptm_file, 'file')
            paratest = readmatrix(paratest_file);
            parainfertest = readmatrix(parainfertest_file);
            for k = 1:5
                y_true = paratest(:, k);
                y_pred = parainfertest(:, k);
                
                % Perform linear regression
                X = [ones(length(y_pred), 1), y_pred];  % Add a column of ones for the intercept
                [b, ~, ~, ~, stats] = regress(y_true, X);
                
                % Store R² value
                R2_values(i, k) = stats(1);
            end
        else
            fprintf('Skipping Job %d: Missing one or more files.\n', jobNumber);
        end
    end
    
    % Convert R2 values to a table and merge with existing dataTable
    R2_table = array2table(R2_values, 'VariableNames', R2_columnNames);
    dataTable_test_2D_1248kdata = [dataTable_test_2D_1248kdata, R2_table];  % Append R2 values to the original table
    dataTable_test_2D_1248kdata{6,9}=0;
    
    dataTable_test_2D_1248kdata.Mean_R2=mean(dataTable_test_2D_1248kdata{:,8:12},2);
    
    
    dataTable_test_2D_10kdata=dataTable_2D_10kdata;
    R2_columnNames ={'Sfal', 'Pxa', 'Sffb', 'PXFCR', 'Fb'};
    numFiles = 16;  % Number of jobs
    R2_values = nan(numFiles, 5);  % 5columns for 5 parameters
    for i = 1:numFiles
        jobNumber = dataTable_test_2D_10kdata.JobNumber(i);
        paratest_file = sprintf('para_and_parainfer/paratest_Mvlognormaldirect_%d.csv', jobNumber);
        parainfertest_file = sprintf('para_and_parainfer/parainfertest_Mvlognormaldirect_%d.csv', jobNumber);
        if exist(paratest_file, 'file') && exist(parainfertest_file, 'file')% && exist(parainferfromoptm_file, 'file')
            paratest = readmatrix(paratest_file);
            parainfertest = readmatrix(parainfertest_file);
            for k = 1:5
                y_true = paratest(:, k);y_pred = parainfertest(:, k);
                X = [ones(length(y_pred), 1), y_pred];  % Add a column of ones for the intercept
                [b, ~, ~, ~, stats] = regress(y_true, X);
                R2_values(i, k) = stats(1);
            end
        else
            fprintf('Skipping Job %d: Missing one or more files.\n', jobNumber);
        end
    end
    
    % Convert R2 values to a table and merge with existing dataTable
    R2_table = array2table(R2_values, 'VariableNames', R2_columnNames);
    dataTable_test_2D_10kdata = [dataTable_test_2D_10kdata, R2_table];  % Append R2 values to the original table
    dataTable_test_2D_10kdata.Mean_R2=mean(dataTable_test_2D_10kdata{:,8:12},2);
    
    
    if iscell(dataTable_test_2D_1248kdata.ntrain)
        dataTable_test_2D_1248kdata.ntrain = cell2mat(dataTable_test_2D_1248kdata.ntrain);
    end
    if iscell(dataTable_test_2D_10kdata.ntrain)
        dataTable_test_2D_10kdata.ntrain = cell2mat(dataTable_test_2D_10kdata.ntrain);
    end
    
    dataTable_test_2D_124810kdata=[dataTable_test_2D_1248kdata;dataTable_test_2D_10kdata([1:4,9:2:15],:)];
    

 
    paratrue=importdata('FX2D25_paraoptm.csv');
    chosensubjs=[1:2,4:10,12:17,19:24];
    
    dataTable_optm_2D_10kdata=dataTable_2D_10kdata;
    R2_true_vs_optm_ColumnNames = {'Sfal', 'Pxa', 'Sffb', 'PXFCR', 'Fb'};
    
    numFiles = 16;  % Number of jobs
    R2_true_vs_optm = nan(numFiles, 5);  % 6 columns for 6 parameters

    for i = 1:numFiles
        jobNumber = dataTable_optm_2D_10kdata.JobNumber(i);
        parainferfromoptm_file = sprintf('para_and_parainfer/parainferfromoptm_Mvlognormaldirect_%d.csv', jobNumber);
    
        if exist(parainferfromoptm_file, 'file')
            parainferfromoptm = readmatrix(parainferfromoptm_file);
            y_true = paratrue(chosensubjs, 1:5);  % Take only first 6 columns
            y_pred = parainferfromoptm(chosensubjs, 1:5);  % Match rows with `paratrue_filtered`
            for k = 1:5
                X = [ones(length(y_pred(:, k)), 1), y_pred(:, k)];  % Add a column of ones for the intercept
                [b, ~, ~, ~, stats] = regress(y_true(:, k), X);
                R2_true_vs_optm(i, k) = stats(1);
            end
        else
            fprintf('Skipping Job %d: Missing parainferfromoptm file.\n', jobNumber);
        end
    end
    
    R2_true_vs_optm_table = array2table(R2_true_vs_optm, 'VariableNames', R2_true_vs_optm_ColumnNames);
    dataTable_optm_2D_10kdata = [dataTable_optm_2D_10kdata, R2_true_vs_optm_table];  % Append new R2 values to the original table
    dataTable_optm_2D_10kdata.Mean_R2=mean(dataTable_optm_2D_10kdata{:,8:12},2);
    
    
    dataTable_optm_2D_1248kdata=dataTable_2D_1248kdata;
    
    numFiles = 32;  % Number of jobs
    R2_true_vs_optm = nan(numFiles, 5);  % 6 columns for 6 parameters
    
    for i = 1:numFiles
        jobNumber = dataTable_optm_2D_1248kdata.JobNumber(i);
        parainferfromoptm_file = sprintf('para_and_parainfer/parainferfromoptm_Mvlognormaldirect_%d.csv', jobNumber);
    
        if exist(parainferfromoptm_file, 'file')
            parainferfromoptm = readmatrix(parainferfromoptm_file);
            y_true = paratrue(chosensubjs, 1:5);  % Take only first 6 columns
            y_pred = parainferfromoptm(chosensubjs, 1:5);  % Match rows with `paratrue_filtered`
            for k = 1:5
                X = [ones(length(y_pred(:, k)), 1), y_pred(:, k)];  % Add a column of ones for the intercept
                [b, ~, ~, ~, stats] = regress(y_true(:, k), X);
                R2_true_vs_optm(i, k) = stats(1);
    
            end
        else
            fprintf('Skipping Job %d: Missing parainferfromoptm file.\n', jobNumber);
        end
    end
    
    R2_true_vs_optm_table = array2table(R2_true_vs_optm, 'VariableNames', R2_true_vs_optm_ColumnNames);
    dataTable_optm_2D_1248kdata = [dataTable_optm_2D_1248kdata, R2_true_vs_optm_table];  % Append new R2 values to the original table
    dataTable_optm_2D_1248kdata.Mean_R2=mean(dataTable_optm_2D_1248kdata{:,8:12},2);
    
    
    if iscell(dataTable_optm_2D_1248kdata.ntrain)
        dataTable_optm_2D_1248kdata.ntrain = cell2mat(dataTable_optm_2D_1248kdata.ntrain);
    end
    if iscell(dataTable_optm_2D_10kdata.ntrain)
        dataTable_optm_2D_10kdata.ntrain = cell2mat(dataTable_optm_2D_10kdata.ntrain);
    end
    dataTable_optm_2D_124810kdata=[dataTable_optm_2D_1248kdata;dataTable_optm_2D_10kdata([1:4,9:2:15],:)];

    %%% Finally, one can run
    writetable(dataTable_1248kdata,'datatables/dataTable_1248kdata_forFigureP.csv');
    %%% writetable(dataTable_test_2D_124810kdata,'datatables/dataTable_test_2D_124810kdata.csv');
    %%% and 
    %%% writetable(dataTable_optm_2D_124810kdata,'datatables/dataTable_optm_2D_124810kdata.csv');
    %%% to save them.
end

%% makefile02 %%%How to make:dataTable_test_2D_500kdata.csv
for kk=1:1
    % 1) List checkpoint files
    F  = dir('model CKPT FILES/250203or250131/*.ckpt');
    fn = {F.name}';
    N  = numel(fn);
    
    % 2) Parse metadata in one pass
    job   = str2double( ...
      regexp(fn, '(?<=_)(?:250203\d{2}|250131\d{2})(?=_)', 'match', 'once') );
    dim   = repmat({'Unknown'}, N, 1);
    dim(contains(fn, 'FX2DNN'))  = {'2D'};
    dim(contains(fn, 'GFX3DNN')) = {'3D'};
    npara = str2double( ...
      regexp(fn, '(?<=_npara)\d+(?=_)', 'match', 'once') );
    ntrain = repmat(500000, N, 1);
    feat  = repmat({'Unknown'}, N, 1);
    feat(contains(fn,'noFE'))         = {'fta_noFE'};
    feat(contains(fn,'catonlytGIF'))  = {'ftb_catonlytGIF'};
    feat(contains(fn,'twithreciGIF')) = {'ftc_twithreciGIF'};
    feat(contains(fn,'fullreci'))     = {'ftd_fullreci'};
    act   = repmat({'Unknown'}, N, 1);
    act(contains(fn,'_relu_'))   = {'relu'};
    act(contains(fn,'_tanh01_')) = {'tanh01'};
    val   = str2double( ...
      regexp(fn, '(?<=-val_loss=)[0-9]*\.?[0-9]+', 'match', 'once') );
    
    % 3) Build and sort table, keep top 35
    T = table(job, dim, npara, ntrain, feat, act, val, ...
        'VariableNames', ...
        {'JobNumber','Dimension','npara','ntrain','FeatureType','Activation','ValLoss'});
    T = sortrows(T, ...
        {'JobNumber','Dimension','npara','ntrain','FeatureType','Activation'});
    T = T(1:35, :);
    
    % 4) Select the 2D subset
    T2D = T(strcmp(T.Dimension,'2D'), :);
    
    % 5) Compute R² for each of the 5 parameters
    M  = height(T2D);
    R2 = nan(M, 5);
    for i = 1:M
        j = T2D.JobNumber(i);
        f1 = sprintf('para_and_parainfer/paratest_Mvlognormaldirect_%d.csv', j);
        f2 = sprintf('para_and_parainfer/parainfertest_Mvlognormaldirect_%d.csv', j);
        if isfile(f1) && isfile(f2)
            Y  = readmatrix(f1);
            Yp = readmatrix(f2);
            for k = 1:5
                [~,~,~,~,stats] = regress(Y(:,k), [ones(size(Yp,1),1), Yp(:,k)]);
                R2(i,k) = stats(1);
            end
        end
    end
    T2D = [T2D, array2table(R2, ...
        'VariableNames',{'R2_Sfal','R2_Pxa','R2_Sffb','R2_PXFCR','R2_Fb'})];
    
    % 6) Pick your final rows and add a Mean_R2 column
    sel = [1:2:7, 9:12];
    dataTable_test_2D_500kdata = T2D(sel, :);
    dataTable_test_2D_500kdata.Mean_R2 = ...
        mean(dataTable_test_2D_500kdata{:, 8:12}, 2);
    
    % % % writetable(dataTable_test_2D_500kdata,'datatables/dataTable_test_2D_500kdata.csv');
end
%% makefile03 %%%How to make:dataTable_optm_2D_500kdata.csv
for kk=1:1
    % 1) Gather and parse checkpoint filenames
    F   = dir('model CKPT FILES/250203or250131/*.ckpt');
    fn  = {F.name}';
    N   = numel(fn);
    
    job   = str2double( ...
        regexp(fn, '(?<=_)(?:250203\d{2}|250131\d{2})(?=_)', 'match', 'once') );
    dim   = repmat({'Unknown'}, N, 1);
    dim(contains(fn,'FX2DNN'))  = {'2D'};
    dim(contains(fn,'GFX3DNN')) = {'3D'};
    npara = str2double( ...
        regexp(fn, '(?<=_npara)\d+(?=_)', 'match', 'once') );
    ntrain = repmat(500000, N, 1);
    feat  = repmat({'Unknown'}, N, 1);
    feat(contains(fn,'noFE'))         = {'fta_noFE'};
    feat(contains(fn,'catonlytGIF'))  = {'ftb_catonlytGIF'};
    feat(contains(fn,'twithreciGIF')) = {'ftc_twithreciGIF'};
    feat(contains(fn,'fullreci'))     = {'ftd_fullreci'};
    act   = repmat({'Unknown'}, N, 1);
    act(contains(fn,'_relu_'))   = {'relu'};
    act(contains(fn,'_tanh01_')) = {'tanh01'};
    val   = str2double( ...
        regexp(fn, '(?<=-val_loss=)[0-9]*\.?[0-9]+', 'match', 'once') );
    
    % 2) Build and trim to first 35 entries
    T = table(job, dim, npara, ntrain, feat, act, val, ...
        'VariableNames',{'JobNumber','Dimension','npara','ntrain', ...
                         'FeatureType','Activation','ValLoss'});
    T = sortrows(T, {'JobNumber','Dimension','npara','ntrain', ...
                     'FeatureType','Activation'});
    T = T(1:35,:);
    
    % 3) Extract the 2D subset
    T2D = T(strcmp(T.Dimension,'2D'), :);
    
    % 4) Load ground-truth parameters and select subjects
    paratrue = importdata('FX2D25_paraoptm.csv');
    chs      = [1:2,4:10,12:17,19:24];
    Yt       = paratrue(chs,1:5);
    
    % 5) Compute R² between true and optimized for each job
    M     = height(T2D);
    R2opt = nan(M,5);
    for i = 1:M
        j    = T2D.JobNumber(i);
        f    = sprintf('para_and_parainfer/parainferfromoptm_Mvlognormaldirect_%d.csv', j);
        if isfile(f)
            Yp = readmatrix(f);
            Yp = Yp(chs,1:5);
            for k = 1:5
                [~,~,~,~,stats] = regress(Yt(:,k), [ones(size(Yp,1),1), Yp(:,k)]);
                R2opt(i,k)      = stats(1);
            end
        end
    end
    
    % 6) Append R² columns and compute final selection
    colNames = {'R2_Sfal','R2_Pxa','R2_Sffb','R2_PXFCR','R2_Fb'};
    T2D = [T2D, array2table(R2opt, 'VariableNames',colNames)];
    
    sel = [1:2:7, 9:12];
    dataTable_optm_2D_500kdata = T2D(sel, :);
    dataTable_optm_2D_500kdata.Mean_R2 = ...
        mean(dataTable_optm_2D_500kdata{:,colNames}, 2);
    
    % % % writetable(dataTable_optm_2D_500kdata,'datatables/dataTable_optm_2D_500kdata.csv');

end


%% makefile04 %%% How to make: 'dataTable_test_3D_124810kdata.csv' 
% and 'dataTable_optm_3D_124810kdata.csv' 

for kk=1:1
    files = dir('model CKPT files/250207/*.ckpt');
    names = {files.name};
    n = numel(files);
    
    % JobNumber
    tok = regexp(names, '_(250207\d{3})_', 'tokens');
    JobNumber = NaN(n,1);
    mask = ~cellfun(@isempty, tok);
    JobNumber(mask) = cellfun(@(c) str2double(c{1}{1}), tok(mask));
    
    % Dimension
    Dimension = repmat({'Unknown'}, n,1);
    Dimension(cellfun(@(s) contains(s,'FX2DNN'), names)) = {'2D'};
    Dimension(cellfun(@(s) contains(s,'GFX3DNN'), names)) = {'3D'};
    
    % npara
    tok = regexp(names, '_npara(\d+)_', 'tokens');
    npara = NaN(n,1);
    mask = ~cellfun(@isempty, tok);
    npara(mask) = cellfun(@(c) str2double(c{1}{1}), tok(mask));
    
    % ntrain
    baseID = 250207033;
    grp = floor((JobNumber - baseID)/8);
    ntrain = 1000 * 2 .^ mod(grp,4);
    
    % FeatureType
    keys = {'noFE','catonlytGIF','twithreciGIF','fullreci'};
    vals = {'fta_noFE','ftb_catonlytGIF','ftc_twithreciGIF','ftd_fullreci'};
    FeatureType = repmat({'Unknown'}, n,1);
    for k = 1:numel(keys)
        idx = contains(names, keys{k});
        FeatureType(idx) = vals(k);
    end
    
    % Activation
    Activation = repmat({'Unknown'}, n,1);
    Activation(contains(names,'_relu_'))   = {'relu'};
    Activation(contains(names,'_tanh01_')) = {'tanh01'};
    
    % ValLoss
    tok = regexp(names, '-val_loss=([0-9]*\.?[0-9]+)', 'tokens');
    ValLoss = NaN(n,1);
    mask = ~cellfun(@isempty, tok);
    ValLoss(mask) = cellfun(@(c) str2double(c{1}{1}), tok(mask));
    
    % Assemble and sort
    dataTable_1248kdata = cell2table([ ...
        num2cell(JobNumber), Dimension, ...
        num2cell(npara), num2cell(ntrain), ...
        FeatureType, Activation, ...
        num2cell(ValLoss) ], ...
      'VariableNames', {'JobNumber','Dimension','npara','ntrain','FeatureType','Activation','ValLoss'});
    dataTable_1248kdata = sortrows( ...
      dataTable_1248kdata, ...
      {'JobNumber','Dimension','npara','FeatureType','Activation'});
    
    dataTable_3D_1248kdata = dataTable_1248kdata(33:64,:);
    dataTable_2D_1248kdata = dataTable_1248kdata(1:32,:);
    
    %%% Section: Compute R² for test set (1248k data)
    dataTable_test_3D_1248kdata = dataTable_3D_1248kdata;
    m = height(dataTable_test_3D_1248kdata);
    R2 = nan(m,6);
    for i = 1:m
        jid = dataTable_test_3D_1248kdata.JobNumber(i);
        f1 = sprintf('para_and_parainfer/paratest_Mvlognormaldirect_%d.csv',   jid);
        f2 = sprintf('para_and_parainfer/parainfertest_Mvlognormaldirect_%d.csv',jid);
        if exist(f1,'file') && exist(f2,'file')
            Ytrue = readmatrix(f1);
            Ypred = readmatrix(f2);
            for k = 1:6
                X = [ones(size(Ypred,1),1), Ypred(:,k)];
                [~,~,~,~,st] = regress(Ytrue(:,k), X);
                R2(i,k) = st(1);
            end
        else
            fprintf('Skipping Job %d: Missing files.\n', jid);
        end
    end
    R2_table = array2table(R2, ...
      'VariableNames', {'R2_SI','R2_CX','R2_SG','R2_X2','R2_CF','R2_L2'});
    dataTable_test_3D_1248kdata = [dataTable_test_3D_1248kdata, R2_table];
    dataTable_test_3D_1248kdata.Mean_R2 = mean(dataTable_test_3D_1248kdata{:,8:13},2);
    
    %%% Section: Compute R² for optimization set (1248k data)
    % Load true parameters
    GIF0      = importdata('25FSIGT_GIF0.csv');
    all6para  = importdata('optimizedparameters_3Dmodel.csv');
    paratrue=[all6para, GIF0(:,[1,3])];
    chosensubjs = [1:2,4:10,12:17,19:24];
    
    dataTable_optm_3D_1248kdata = dataTable_3D_1248kdata;
    R2o = nan(m,6);
    for i = 1:m
        jid = dataTable_optm_3D_1248kdata.JobNumber(i);
        f  = sprintf('para_and_parainfer/parainferfromoptm_Mvlognormaldirect_%d.csv', jid);
        if exist(f,'file')
            Ypred = readmatrix(f);
            Yt = paratrue(chosensubjs,1:6);
            Yp = Ypred(chosensubjs, 1:6);
            for k = 1:6
                X = [ones(size(Yp,1),1), Yp(:,k)];
                [~,~,~,~,st] = regress(Yt(:,k), X);
                R2o(i,k) = st(1);
            end
        else
            fprintf('Skipping Job %d: Missing file.\n', jid);
        end
    end
    R2o_table = array2table(R2o, ...
      'VariableNames', {'R2true_SI','R2true_CX','R2true_SG','R2true_X2','R2true_CF','R2true_L2'});
    dataTable_optm_3D_1248kdata = [dataTable_optm_3D_1248kdata, R2o_table];
    dataTable_optm_3D_1248kdata.Mean_R2 = mean(dataTable_optm_3D_1248kdata{:,8:13},2);
    
    %%% Section: Parse CKPT files for 250205 (10k data)
    files = dir('model CKPT files/250205/*.ckpt');
    names = {files.name};
    n = numel(files);
    
    tok = regexp(names, '_(250205\d{3})_', 'tokens');
    JobNumber = NaN(n,1);
    mask = ~cellfun(@isempty, tok);
    JobNumber(mask) = cellfun(@(c) str2double(c{1}{1}), tok(mask));
    
    Dimension = repmat({'Unknown'}, n,1);
    Dimension(cellfun(@(s) contains(s,'FX2DNN'), names)) = {'2D'};
    Dimension(cellfun(@(s) contains(s,'GFX3DNN'), names)) = {'3D'};
    
    tok = regexp(names, '_npara(\d+)_', 'tokens');
    npara = NaN(n,1);
    mask = ~cellfun(@isempty, tok);
    npara(mask) = cellfun(@(c) str2double(c{1}{1}), tok(mask));
    
    ntrain = repmat(10000, n,1);
    
    featType = repmat({'Unknown'}, n,1);
    for k = 1:numel(keys)
        featType(contains(names, keys{k})) = vals(k);
    end
    
    act = repmat({'Unknown'}, n,1);
    act(contains(names,'_relu_'))   = {'relu'};
    act(contains(names,'_tanh01_')) = {'tanh01'};
    
    tok = regexp(names, '-val_loss=([0-9]*\.?[0-9]+)', 'tokens');
    ValLoss = NaN(n,1);
    mask = ~cellfun(@isempty, tok);
    ValLoss(mask) = cellfun(@(c) str2double(c{1}{1}), tok(mask));
    
    dataTable_10kdata = cell2table([ ...
        num2cell(JobNumber), Dimension, ...
        num2cell(npara), num2cell(ntrain), ...
        featType, act, ...
        num2cell(ValLoss) ], ...
      'VariableNames', {'JobNumber','Dimension','npara','ntrain','FeatureType','Activation','ValLoss'});
    dataTable_10kdata = sortrows( ...
      dataTable_10kdata, ...
      {'JobNumber','Dimension','npara','ntrain','FeatureType','Activation'});
    
    dataTable_3D_10kdata = dataTable_10kdata(1:24,:);
    dataTable_2D_10kdata = dataTable_10kdata(25:40,:);
    
    %%% Section: Compute R² for test set (10k data)
    dataTable_test_3D_10kdata = dataTable_3D_10kdata;
    m2 = height(dataTable_test_3D_10kdata);
    R2 = nan(m2,6);
    for i = 1:m2
        jid = dataTable_test_3D_10kdata.JobNumber(i);
        f1 = sprintf('para_and_parainfer/paratest_Mvlognormaldirect_%d.csv',   jid);
        f2 = sprintf('para_and_parainfer/parainfertest_Mvlognormaldirect_%d.csv',jid);
        if exist(f1,'file') && exist(f2,'file')
            Ytrue = readmatrix(f1);
            Ypred = readmatrix(f2);
            for k = 1:6
                X = [ones(size(Ypred,1),1), Ypred(:,k)];
                [~,~,~,~,st] = regress(Ytrue(:,k), X);
                R2(i,k) = st(1);
            end
        else
            fprintf('Skipping Job %d: Missing files.\n', jid);
        end
    end
    R2_table = array2table(R2, ...
      'VariableNames', {'R2_SI','R2_CX','R2_SG','R2_X2','R2_CF','R2_L2'});
    dataTable_test_3D_10kdata = [dataTable_test_3D_10kdata, R2_table];
    dataTable_test_3D_10kdata.Mean_R2 = mean(dataTable_test_3D_10kdata{:,8:13},2);
    
    %%% Section: Compute R² for optimization set (10k data)
    dataTable_optm_3D_10kdata = dataTable_3D_10kdata;
    R2o = nan(m2,6);
    for i = 1:m2
        jid = dataTable_optm_3D_10kdata.JobNumber(i);
        f  = sprintf('para_and_parainfer/parainferfromoptm_Mvlognormaldirect_%d.csv', jid);
        if exist(f,'file')
            Ypred = readmatrix(f);
            Yt = paratrue(chosensubjs,1:6);
            Yp = Ypred(chosensubjs,1:6);
            for k = 1:6
                X = [ones(size(Yp,1),1), Yp(:,k)];
                [~,~,~,~,st] = regress(Yt(:,k), X);
                R2o(i,k) = st(1);
            end
        else
            fprintf('Skipping Job %d: Missing file.\n', jid);
        end
    end
    R2o_table = array2table(R2o, ...
      'VariableNames', {'R2true_SI','R2true_CX','R2true_SG','R2true_X2','R2true_CF','R2true_L2'});
    dataTable_optm_3D_10kdata = [dataTable_optm_3D_10kdata, R2o_table];
    dataTable_optm_3D_10kdata.Mean_R2 = mean(dataTable_optm_3D_10kdata{:,8:13},2);
    
    %%% Section: Merge 1248k and 10k results
    dataTable_test_3D_124810kdata = [
        dataTable_test_3D_1248kdata; 
        dataTable_test_3D_10kdata(dataTable_test_3D_10kdata.npara==6,:)
    ];
    dataTable_optm_3D_124810kdata = [
        dataTable_optm_3D_1248kdata; 
        dataTable_optm_3D_10kdata(dataTable_optm_3D_10kdata.npara==6,:)
    ];


     % % % writetable(dataTable_test_3D_124810kdata,'datatables/dataTable_test_3D_124810kdata.csv');
     % % % writetable(dataTable_optm_3D_124810kdata,'datatables/dataTable_optm_3D_124810kdata.csv');
end


 

%% makefile05 %%% How to make: 'dataTable_test_3D_500kdata.csv' and
%%% 'dataTable_optm_3D_500kdata.csv'

for kk=1:1
    files = dir('model CKPT files/250203or250131/*.ckpt');
    names = {files.name};
    n = numel(names);
    
    tok = regexp(names,'_(250203\d{2}|250131\d{2})_','tokens');
    JobNumber = NaN(n,1);
    mask = ~cellfun(@isempty,tok);
    JobNumber(mask) = cellfun(@(c)str2double(c{1}{1}),tok(mask));
    
    Dimension = repmat({'Unknown'},n,1);
    Dimension(contains(names,'FX2DNN'))  = {'2D'};
    Dimension(contains(names,'GFX3DNN')) = {'3D'};
    
    tok = regexp(names,'_npara(\d+)_','tokens');
    npara = NaN(n,1);
    mask = ~cellfun(@isempty,tok);
    npara(mask) = cellfun(@(c)str2double(c{1}{1}),tok(mask));
    
    ntrain = repmat(500000,n,1);
    
    keys = {'noFE','catonlytGIF','twithreciGIF','fullreci'};
    vals = {'fta_noFE','ftb_catonlytGIF','ftc_twithreciGIF','ftd_fullreci'};
    FeatureType = repmat({'Unknown'},n,1);
    for k = 1:numel(keys)
        FeatureType(contains(names,keys{k})) = vals(k);
    end
    
    Activation = repmat({'Unknown'},n,1);
    Activation(contains(names,'_relu_'))   = {'relu'};
    Activation(contains(names,'_tanh01_')) = {'tanh01'};
    
    tok = regexp(names,'-val_loss=([0-9]*\.?[0-9]+)','tokens');
    ValLoss = NaN(n,1);
    mask = ~cellfun(@isempty,tok);
    ValLoss(mask) = cellfun(@(c)str2double(c{1}{1}),tok(mask));
    
    dataTable = cell2table([num2cell(JobNumber),Dimension, num2cell(npara), num2cell(ntrain), FeatureType, Activation, num2cell(ValLoss)], ...
        'VariableNames',{'JobNumber','Dimension','npara','ntrain','FeatureType','Activation','ValLoss'});
    dataTable = sortrows(dataTable,{'JobNumber','Dimension','npara','ntrain','FeatureType','Activation'});
    dataTable = dataTable(1:35,:);
    dataTable_3D = dataTable(17:34,:);
    dataTable_2D = dataTable(1:16,:);
    
    %%% Section: Compute R² for test set (250203 or 250131)
    dataTable_test_3D = dataTable_3D;
    m = height(dataTable_test_3D);
    R2 = nan(m,6);
    for i = 1:m
        jid = dataTable_test_3D.JobNumber(i);
        f1 = sprintf('para_and_parainfer/paratest_Mvlognormaldirect_%d.csv',jid);
        f2 = sprintf('para_and_parainfer/parainfertest_Mvlognormaldirect_%d.csv',jid);
        if exist(f1,'file') && exist(f2,'file')
            Yt = readmatrix(f1);
            Yp = readmatrix(f2);
            for k = 1:6
                X = [ones(size(Yp,1),1), Yp(:,k)];
                [~,~,~,~,st] = regress(Yt(:,k),X);
                R2(i,k) = st(1);
            end
        end
    end
    R2_table = array2table(R2,'VariableNames',{'R2_SI','R2_CX','R2_SG','R2_X2','R2_CF','R2_L2'});
    dataTable_test_3D = [dataTable_test_3D, R2_table];
    dataTable_test_3D.Mean_R2 = mean(dataTable_test_3D{:,8:13},2);
    
    %%% Section: Compute R² for optimization set (250203 or 250131)
    GIF0      = importdata('25FSIGT_GIF0.csv');
    all6para  = importdata('optimizedparameters_3Dmodel.csv');
    paratrue  = [all6para, GIF0(:,[1,3])];
    chosensubjs = [1:2,4:10,12:17,19:24];
    
    dataTable_optm_3D = dataTable_3D;
    R2o = nan(m,6);
    for i = 1:m
        jid = dataTable_optm_3D.JobNumber(i);
        f  = sprintf('para_and_parainfer/parainferfromoptm_Mvlognormaldirect_%d.csv',jid);
        if exist(f,'file')
            Yp = readmatrix(f);
            Yt = paratrue(chosensubjs,1:6);
            Yp = Yp(chosensubjs,1:6);
            for k = 1:6
                X = [ones(size(Yp,1),1), Yp(:,k)];
                [~,~,~,~,st] = regress(Yt(:,k),X);
                R2o(i,k) = st(1);
            end
        end
    end
    R2o_table = array2table(R2o,'VariableNames',{'R2true_SI','R2true_CX','R2true_SG','R2true_X2','R2true_CF','R2true_L2'});
    dataTable_optm_3D = [dataTable_optm_3D, R2o_table];
    dataTable_optm_3D.Mean_R2 = mean(dataTable_optm_3D{:,8:13},2);
    
    %%% Section: Parse CKPT files for 250211
    files = dir('model CKPT files/250211/*.ckpt');
    names = {files.name};
    n = numel(names);
    
    tok = regexp(names,'_(250211\d{3})_','tokens');
    JobNumber = NaN(n,1);
    mask = ~cellfun(@isempty,tok);
    JobNumber(mask) = cellfun(@(c)str2double(c{1}{1}),tok(mask));
    
    Dimension = repmat({'Unknown'},n,1);
    Dimension(contains(names,'FX2DNN'))  = {'2D'};
    Dimension(contains(names,'GFX3DNN')) = {'3D'};
    
    tok = regexp(names,'_npara(\d+)_','tokens');
    npara = NaN(n,1);
    mask = ~cellfun(@isempty,tok);
    npara(mask) = cellfun(@(c)str2double(c{1}{1}),tok(mask));
    
    ntrain = repmat(500000,n,1);
    
    FeatureType = repmat({'Unknown'},n,1);
    for k = 1:numel(keys)
        FeatureType(contains(names,keys{k})) = vals(k);
    end
    
    Activation = repmat({'Unknown'},n,1);
    Activation(contains(names,'_relu_'))   = {'relu'};
    Activation(contains(names,'_tanh01_')) = {'tanh01'};
    
    tok = regexp(names,'-val_loss=([0-9]*\.?[0-9]+)','tokens');
    ValLoss = NaN(n,1);
    mask = ~cellfun(@isempty,tok);
    ValLoss(mask) = cellfun(@(c)str2double(c{1}{1}),tok(mask));
    
    dataTable_3D_500kdata_fullreci = cell2table([num2cell(JobNumber),Dimension, num2cell(npara), num2cell(ntrain), FeatureType, Activation, num2cell(ValLoss)], ...
        'VariableNames',{'JobNumber','Dimension','npara','ntrain','FeatureType','Activation','ValLoss'});
    dataTable_3D_500kdata_fullreci = sortrows(dataTable_3D_500kdata_fullreci,{'JobNumber','Dimension','npara','FeatureType','Activation'});
    
    %%% Section: Compute R² for test set (250211)
    dataTable_test_3D_500kdata_fullreci = dataTable_3D_500kdata_fullreci(1:6,:);
    m2 = height(dataTable_test_3D_500kdata_fullreci);
    R2 = nan(m2,6);
    for i = 1:m2
        jid = dataTable_test_3D_500kdata_fullreci.JobNumber(i);
        f1 = sprintf('para_and_parainfer/paratest_Mvlognormaldirect_%d.csv',jid);
        f2 = sprintf('para_and_parainfer/parainfertest_Mvlognormaldirect_%d.csv',jid);
        if exist(f1,'file') && exist(f2,'file')
            Yt = readmatrix(f1);
            Yp = readmatrix(f2);
            for k = 1:6
                X = [ones(size(Yp,1),1), Yp(:,k)];
                [~,~,~,~,st] = regress(Yt(:,k),X);
                R2(i,k) = st(1);
            end
        end
    end
    R2_table = array2table(R2,'VariableNames',{'R2_SI','R2_CX','R2_SG','R2_X2','R2_CF','R2_L2'});
    dataTable_test_3D_500kdata_fullreci = [dataTable_test_3D_500kdata_fullreci, R2_table];
    dataTable_test_3D_500kdata_fullreci.Mean_R2 = mean(dataTable_test_3D_500kdata_fullreci{:,8:13},2);
    
    %%% Section: Compute R² for optimization set (250211)
    dataTable_optm_3D_500kdata_fullreci = dataTable_3D_500kdata_fullreci(1:6,:);
    R2o = nan(m2,6);
    for i = 1:m2
        jid = dataTable_optm_3D_500kdata_fullreci.JobNumber(i);
        f  = sprintf('para_and_parainfer/parainferfromoptm_Mvlognormaldirect_%d.csv',jid);
        if exist(f,'file')
            Yp = readmatrix(f);
            Yt = paratrue(chosensubjs,1:6);
            Yp = Yp(chosensubjs,1:6);
            for k = 1:6
                X = [ones(size(Yp,1),1), Yp(:,k)];
                [~,~,~,~,st] = regress(Yt(:,k),X);
                R2o(i,k) = st(1);
            end
        end
    end
    R2o_table = array2table(R2o,'VariableNames',{'R2true_SI','R2true_CX','R2true_SG','R2true_X2','R2true_CF','R2true_L2'});
    dataTable_optm_3D_500kdata_fullreci = [dataTable_optm_3D_500kdata_fullreci, R2o_table];
    dataTable_optm_3D_500kdata_fullreci.Mean_R2 = mean(dataTable_optm_3D_500kdata_fullreci{:,8:13},2);
    
    %%% Section: Merge all results
    dataTable_test_3D_500kdata  = [dataTable_test_3D;  dataTable_test_3D_500kdata_fullreci];
    dataTable_optm_3D_500kdata= [dataTable_optm_3D; dataTable_optm_3D_500kdata_fullreci];

%     writetable(dataTable_test_3D_500kdata,'datatables/dataTable_test_3D_500kdata.csv');
%     writetable(dataTable_optm_3D_500kdata,'datatables/dataTable_optm_3D_500kdata.csv');


end



%%

jobid=250207002;


fn1=sprintf('paratest_Mvlognormaldirect_%d.csv',jobid);

fn2=sprintf('parainfertest_Mvlognormaldirect_%d.csv',jobid);

fn3=sprintf('parainferfromoptm_Mvlognormaldirect_%d.csv',jobid);

paratest_new=importdata(fn1);
parainfertest_new=importdata(fn2);
parainferoptm_new=importdata(fn3);

fn1=sprintf('para_and_parainfer/paratest_Mvlognormaldirect_%d.csv',jobid);
fn2=sprintf('para_and_parainfer/parainfertest_Mvlognormaldirect_%d.csv',jobid);
fn3=sprintf('para_and_parainfer/parainferfromoptm_Mvlognormaldirect_%d.csv',jobid);

paratest_old=importdata(fn1);
parainfertest_old=importdata(fn2);
parainferoptm_old=importdata(fn3);

 

