function [rmsvars lowndx rmstrain rmstest] = run
    filename = 'commodity_prices.csv';
    data = readmatrix(filename);
    data = data(:, 2:end); % remove dates column
    [m, n] = size(data);

    % change empty values to avg of column:
    datanan = isnan(data);
    for col=1:n
        avg=mean(data(:, col), 'omitnan');
        for row=1:m
            if datanan(row, col) == 1
                data(row, col) = avg;
            end
        end
    end
    [rmsvars lowndx] = find_best_variable(data);
    [rmstrain rmstest] = cross_validation(data, lowndx);
    disp("Training RMS errors for each fold:")
    disp(rmstrain)
    disp("Average training RMS error:")
    disp(mean(rmstrain))
    disp("Validation RMS errors for each fold:")
    disp(rmstest)
    disp("Average validation RMS error:")
    disp(mean(rmstest))
    percentdiff = 100*abs((mean(rmstrain) - mean(rmstest)) / mean(rmstest));
    if mean(rmstest) > mean(rmstrain)
        disp("Training RMS error was " + percentdiff + "% less than validation RMS error")
    else
        disp("Training RMS error was " + percentdiff + "% greater than validation RMS error")
    end
    disp("Index of variable with lowest RMS error:")
    disp(lowndx)
    disp("RMS error for each variable as dependent variable:")
    


end

function [rmsvars lowndx] = find_best_variable(data)
    Amat = data;
    % onesVec = ones(m, 1)
    % Amat = [Amat onesVec]
    [m, n] = size(Amat);

    % Compute the RMS errors for linear regression
    dataStand = zscore(Amat); % Amat standardized

    rmsvars = [];
    for col = 1:n
        c = dataStand(:, col); % independent variable
        A = dataStand(:, [1:col-1, col+1:end]); % All columns of data except for c
        w = A\c; % w s.t. A*w ~= c
        err = c - A*w; % error vector
        errRMS = rms(err);
        rmsvars(end + 1) = errRMS;
    end

    [M, lowndx] = min(rmsvars);

    % Plot the results
    A = dataStand(:, [1:lowndx-1, lowndx+1:end]);
    bestC = Amat(:, lowndx);
    cStd = std(bestC);
    cMean = mean(bestC);
    w = A\c;
    predictedCStand = A*w;
    predictedC = predictedCStand * cStd + cMean; % destandardized prediction
    x = 1:m;
    plot(x, bestC, 'b');
    hold on;
    plot(x, predictedC, 'r');
    hold off;
    xlabel('Time (Abitrary units)');
    ylabel('Price');
    %legend('Actual values', 'Predicted values');

end
function [rmstrain rmstest] = cross_validation(data,lowndx)
    [m, n] = size(data);

    % Create Xmat and yvec from the data and the input parameter
    Amat = data(:, [1:lowndx-1, lowndx+1:end]); % Remove lowndx-th column
    n = n - 1; % account for change in size
    yvec = data(:, lowndx); % lowndx-th column
    Xmat = Amat(randperm(m), :); % randomly shuffle rows
    %Xmat = Amat;
 

    % Compute the RMS errors of 5-fold cross-validation
    [rmstrain, rmstest] = mykfold(Xmat, yvec, 9);

end

function [rmstrain,rmstest]=mykfold(Xmat, yvec, k_in)

    % Problem size
    M = size(Xmat, 1);

    % Set the number of folds; must be 1<k<M
    if nargin >= 3 & ~isempty(k_in)
        k = max(min(round(k_in), M-1), 2);
    else
        k = 5;
    end
    
    data = Xmat;
    dataStandardized = zscore(data);
    c = yvec;
    % record c standard deviation and mean before standardizing so that we
    % can destandardize
    cStd = std(c);
    cMean = mean(c);
    cStandardized = zscore(c);

    % initialize variables to store folds
    dataStandFolds = {};
    cStandFolds = {};
    cFolds = {};
    place = 1;
    for i=1:k
        stop = place + floor(M / k);
        if stop > M
            stop = M;
        end
        dataStandFolds{end+1}=dataStandardized(place : stop, :);
        cStandFolds{end+1}=cStandardized(place : stop, 1);
        cFolds{end+1}=c(place : stop, 1);
        place = stop + 1;
    end
    
    % Initialize the return variables
    rmstrain = zeros(1, k);
    rmstest  = zeros(1, k);

    % Process each fold
    for ix=1:k
        % initialize empty variables to store folds for training
        dataStand_train = [];
        cStand_train = [];
        c_train = [];

        % for all folds except ix-th
        for i=[1:ix-1, ix+1:k]
            % add folds to training data
            dataStand_train = [dataStand_train; dataStandFolds{i}];
            cStand_train = [cStand_train; cStandFolds{i}];
            c_train = [c_train; cFolds{i}];
        end
        
        % we are testing on ix-th fold
        dataStand_test = dataStandFolds{ix};
        cStand_test = cStandFolds{ix};
        c_test = cFolds{ix};
        
        wvec = dataStand_train\cStand_train; %use training data to find w

        cTrainPredictionStandardized = dataStand_train * wvec; % Standardized prediction of c from training data
        cTrainPrediction = cTrainPredictionStandardized*cStd + cMean; % destandardize prediction
        rmstrain(ix) = rms(cTrainPrediction - c_train); % Calculate RMSE

        cTestPredictionStandardized = dataStand_test * wvec; % Standardized prediction of c from test data
        cTestPrediction = cTestPredictionStandardized*cStd + cMean; % destandardize prediction
        rmstest(ix) = rms(cTestPrediction - c_test); % Calculate RMSE
    end

end
