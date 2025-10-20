function [X, truth, folds, X_com] = prepareMissingData(datasetName, missingRate)
   
    dataPath = sprintf('dataset\\%s.mat', datasetName);
    load(dataPath); 

    switch datasetName
        case 'COIL20MV'
            for v=1:3
            X{v} = X{v}';
            end
            gt = double(gt);
            truth = gt;
        otherwise
            error('Unsupported dataset: %s', datasetName);
    end

    X_com = X;
    numViews = length(X);
    numSamples = size(X{1}, 1);
    
    if missingRate>(1-1/numViews)
        error('The missing rate exceeds 1-1/numViews');
    end

    foldFile = sprintf('dataset\\%s_folds_%.1f.mat', datasetName, missingRate);

    if exist(foldFile, 'file') == 2
        load(foldFile);  
    else
        folds = ones(numSamples, numViews);
        for i = 1:numViews
            mask = rand(numSamples, 1) > missingRate;
            folds(:, i) = mask;
        end


        validRows = any(folds == 1, 2);
        missingIndices = find(~validRows);
        for idx = missingIndices'
            viewToRecover = randi(numViews);
            folds(idx, viewToRecover) = 1;
        end

        save(foldFile, 'folds');
    end
end
