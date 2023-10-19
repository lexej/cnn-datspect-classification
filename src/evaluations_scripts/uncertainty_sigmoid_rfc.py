import numpy as np
import csv
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def uncertainty_sigmoid(n_splits=1, target_balanced_accuracy=98.0):
    ppmi_flg = True
    mph_flg = True

    # Defaults
    cutoff_natural = 0.5  # natural cutoff on sigmoid output
    natural_cutoff_flg = True  # use natural cutoff

    # Development dataset (n = 2*6*1740 = 20880)
    # Get filenames of CSV tables with sigmoid outputs (one table per random split)
    file_names = ["/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_split4.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_split5.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_split6.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_split7.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_split8.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_split9.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_splittest.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_splittrain.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_splittrain0.csv",
                  "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_baseline_majority/preds_all_splitvalid.csv"
                  ]  # Replace this with the list of file paths to your CSV files

    # Preparations for accuracy
    AUC = np.zeros(n_splits)
    cutoff = np.zeros(n_splits)
    bacc = np.zeros((n_splits, 3))
    acc = np.zeros((n_splits, 3))
    sens = np.zeros((n_splits, 3))
    spec = np.zeros((n_splits, 3))
    ppv = np.zeros((n_splits, 3))
    npv = np.zeros((n_splits, 3))

    # Preparations for inconclusive cases
    percent_incon = np.arange(0.2, 50.2, 0.2)
    percent_incon_max = 20
    ind_percent_incon_max = np.argmin(np.abs(percent_incon - percent_incon_max)) + 1
    n_prop = len(percent_incon)

    lower_bound = np.zeros((n_splits, n_prop))
    upper_bound = np.zeros((n_splits, n_prop))
    observed_percent_incon = np.zeros((n_splits, n_prop))
    bacc_incon = np.zeros((n_splits, n_prop))
    bacc_con = np.zeros((n_splits, n_prop))

    # Loop through the random splits
    for i in range(n_splits):
        # Replace the importCSVdevelopment and other functions with your data loading logic
        # Import the data from CSV files
        sample, majority_vote, prediction = importCSVdevelopment(file_names[i])

        sample = np.array(sample)
        majority_vote = np.array(majority_vote)
        prediction = np.array(prediction)

        # Training
        m = (sample == '1')
        score = -prediction[m]
        true_label = majority_vote[m]

        # Replace the following lines with your calculations for TP, FP, TN, FN, and acc_metrics
        TP, FP, TN, FN = crossTable(-cutoff_natural, score, true_label)
        bacc[i, 0], acc[i, 0], sens[i, 0], spec[i, 0], ppv[i, 0], npv[i, 0] = accMetrics(TP, FP, TN, FN)

        # Validation
        m = (sample == '2')
        score = -prediction[m]
        true_label = majority_vote[m]

        # Replace the following lines with your calculations for AUC, cutoff, and acc_metrics
        AUC[i], cutoff[i] = ROC(score, true_label, False)
        if natural_cutoff_flg:
            cutoff[i] = -cutoff_natural
        TP, FP, TN, FN = crossTable(cutoff[i], score, true_label)
        bacc[i, 1], acc[i, 1], sens[i, 1], spec[i, 1], ppv[i, 1], npv[i, 1] = accMetrics(TP, FP, TN, FN)
        lower_bound[i, :], upper_bound[i, :] = inconInterval(score, cutoff[i], percent_incon)

        # Testing
        m = (sample == '3')
        score = -prediction[m]
        true_label = majority_vote[m]

        # Replace the following lines with your calculations for TP, FP, TN, FN, and acc_metrics
        TP, FP, TN, FN = crossTable(cutoff[i], score, true_label)
        bacc[i, 2], acc[i, 2], sens[i, 2], spec[i, 2], ppv[i, 2], npv[i, 2] = accMetrics(TP, FP, TN, FN)

        # Test inconclusive range in the test set
        m = (sample == '3')
        score = -prediction[m]
        true_label = majority_vote[m]

        # Replace the following lines with your calculations for observed_percent_incon, bacc_incon, and bacc_con
        #   lower_bound and upper_bound -> slice and preserve first dimension 
        observed_percent_incon[i, :], bacc_incon[i, :], bacc_con[i, :] = testInconInterval(score, true_label, cutoff[i], percent_incon, lower_bound[[i], :], upper_bound[[i], :])

    # Average over random splits
    mean_cutoff = -np.mean(cutoff)
    std_cutoff = np.std(cutoff)
    mean_lower_bound = -np.mean(lower_bound, axis=0)
    mean_upper_bound = -np.mean(upper_bound, axis=0)
    std_lower_bound = np.std(lower_bound, axis=0)
    std_upper_bound = np.std(upper_bound, axis=0)
    mean_observed_percent_incon = np.mean(observed_percent_incon, axis=0)
    std_observed_percent_incon = np.std(observed_percent_incon, axis=0)
    mean_bacc_incon = 100 * np.mean(bacc_incon, axis=0, where=~np.isnan(bacc_incon))
    std_bacc_incon = 100 * np.std(bacc_incon, axis=0, where=~np.isnan(bacc_incon))
    mean_bacc_con = 100 * np.mean(bacc_con, axis=0, where=~np.isnan(bacc_con))
    std_bacc_con = 100 * np.std(bacc_con, axis=0, where=~np.isnan(bacc_con))

    
    # Plot lower and upper bound of inconclusive range
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.errorbar(x=percent_incon[:ind_percent_incon_max], 
                y=mean_lower_bound[:ind_percent_incon_max], 
                yerr=std_lower_bound[:ind_percent_incon_max], 
                label='Lower bound',
                marker='o', color='red')
    ax.errorbar(x=percent_incon[:ind_percent_incon_max], 
                y=mean_upper_bound[:ind_percent_incon_max], 
                yerr=std_upper_bound[:ind_percent_incon_max],
                label='Upper bound',
                marker='*', color='blue')

    if natural_cutoff_flg:
        ax.axhline(y=mean_cutoff, label='Cutoff', color='black', linestyle='-')
    else:
        ax.errorbar(x=percent_incon[:ind_percent_incon_max], 
                    y=[mean_cutoff] * ind_percent_incon_max, 
                    yerr=[std_cutoff] * ind_percent_incon_max, 
                    label='Cutoff',
                    color='black', linestyle='-')

    ax.legend()
    ax.set_xlabel('Percent inconclusive cases (%)')
    ax.set_ylabel('Sigmoid')
    #plt.axis([0, percent_incon[ind_percent_incon_max], 0, 1])
    plt.show()

    # Primary quality metric (relF): area under the curve of balanced accuracy in conclusive cases
    # versus proportion of inconclusive cases (scaled to the maximum possible area)
    relF = plotBacc(percent_incon, ind_percent_incon_max, mean_observed_percent_incon, std_observed_percent_incon, mean_bacc_incon, std_bacc_incon, mean_bacc_con, std_bacc_con, 'development')

    # Get proportion of inconclusives at target balanced accuracy in conclusive cases
    ind = np.where(mean_bacc_con >= target_balanced_accuracy)[0]
    percent_incon_at_target = percent_incon[ind[0]]
    mean_lower_bound_at_target = mean_lower_bound[ind[0]]
    mean_upper_bound_at_target = mean_upper_bound[ind[0]]
    std_lower_bound_at_target = std_lower_bound[ind[0]]
    std_upper_bound_at_target = std_upper_bound[ind[0]]

    # Display results
    print('\n\nResults:')
    print(f'\tAUC = {np.mean(AUC):.3f} +/- {np.std(AUC):.3f}')
    print(f'\tCutoff = {mean_cutoff:.3f} +/- {std_cutoff:.3f}')
    print(f'\tbACC: Train = {np.mean(bacc[:, 0]):.3f} +/- {np.std(bacc[:, 0]):.3f}\t'
          f'Valid = {np.mean(bacc[:, 1]):.3f} +/- {np.std(bacc[:, 1]):.3f}\t'
          f'Test = {np.mean(bacc[:, 2]):.3f} +/- {np.std(bacc[:, 2]):.3f}')
    print(f'\tACC: Train = {np.mean(acc[:, 0]):.3f} +/- {np.std(acc[:, 0]):.3f}\t'
          f'Valid = {np.mean(acc[:, 1]):.3f} +/- {np.std(acc[:, 1]):.3f}\t'
          f'Test = {np.mean(acc[:, 2]):.3f} +/- {np.std(acc[:, 2]):.3f}')
    print(f'\tSENS: Train = {np.mean(sens[:, 0]):.3f} +/- {np.std(sens[:, 0]):.3f}\t'
          f'Valid = {np.mean(sens[:, 1]):.3f} +/- {np.std(sens[:, 1]):.3f}\t'
          f'Test = {np.mean(sens[:, 2]):.3f} +/- {np.std(sens[:, 2]):.3f}')
    print(f'\tSPEC: Train = {np.mean(spec[:, 0]):.3f} +/- {np.std(spec[:, 0]):.3f}\t'
          f'Valid = {np.mean(spec[:, 1]):.3f} +/- {np.std(spec[:, 1]):.3f}\t'
          f'Test = {np.mean(spec[:, 2]):.3f} +/- {np.std(spec[:, 2]):.3f}')
    print(f'\tPPV:  Train = {np.mean(ppv[:, 0]):.3f} +/- {np.std(ppv[:, 0]):.3f}\t'
          f'Valid = {np.mean(ppv[:, 1]):.3f} +/- {np.std(ppv[:, 1]):.3f}\t'
          f'Test = {np.mean(ppv[:, 2]):.3f} +/- {np.std(ppv[:, 2]):.3f}')
    print(f'\tNPV:  Train = {np.mean(npv[:, 0]):.3f} +/- {np.std(npv[:, 0]):.3f}\t'
          f'Valid = {np.mean(npv[:, 1]):.3f} +/- {np.std(npv[:, 1]):.3f}\t'
          f'Test = {np.mean(npv[:, 2]):.3f} +/- {np.std(npv[:, 2]):.3f}')
    print(f'\tInconclusives at {target_balanced_accuracy:.1f}% balanced accuracy: {percent_incon_at_target:.1f}%')
    print(f'\tInconclusive SBR range: {mean_lower_bound_at_target:.3f} +/- {std_lower_bound_at_target:.3f} - '
          f'{mean_upper_bound_at_target:.3f} +/- {std_upper_bound_at_target:.3f}')

    if ppmi_flg:
        # PPMI dataset (n = 645)
        # Get filenames of CSV tables with sigmoid outputs (one table per random split)
        ppmi_file_names = [
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_split4.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_split5.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_split6.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_split7.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_split8.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_split9.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_splittest.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_splittrain.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_splittrain0.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_ppmi/preds_baseline_majority_randomization_splitvalid.csv",
        ]

        # Do the work for PPMI dataset
        bacc_incon_ppmi = np.zeros((n_splits, n_prop))
        bacc_con_ppmi = np.zeros((n_splits, n_prop))
        observed_percent_incon_ppmi = np.zeros((n_splits, n_prop))

        for i in range(n_splits):
            ppmi_score, ppmi_true_label = importCSVother(ppmi_file_names[i])

            ppmi_score = np.array(ppmi_score)
            ppmi_true_label = np.array(ppmi_true_label)

            ppmi_score = -ppmi_score
            observed_percent_incon_ppmi[i, :], bacc_incon_ppmi[i, :], bacc_con_ppmi[i, :] = testInconInterval(ppmi_score, ppmi_true_label, cutoff[i], percent_incon, lower_bound[[i], :], upper_bound[[i], :])

        # Average over random splits for PPMI dataset
        mean_observed_percent_incon_ppmi = np.mean(observed_percent_incon_ppmi, axis=0)
        std_observed_percent_incon_ppmi = np.std(observed_percent_incon_ppmi, axis=0)
        mean_bacc_incon_ppmi = 100 * np.mean(bacc_incon_ppmi, axis=0, where=~np.isnan(bacc_incon_ppmi))
        std_bacc_incon_ppmi = 100 * np.std(bacc_incon_ppmi, axis=0, where=~np.isnan(bacc_incon_ppmi))
        mean_bacc_con_ppmi = 100 * np.mean(bacc_con_ppmi, axis=0, where=~np.isnan(bacc_con_ppmi))
        std_bacc_con_ppmi = 100 * np.std(bacc_con_ppmi, axis=0, where=~np.isnan(bacc_con_ppmi))

        # Scaled area under balanced accuracy in cnclusive cases
        relFppmi = plotBacc(percent_incon, ind_percent_incon_max, mean_observed_percent_incon_ppmi, std_observed_percent_incon_ppmi, mean_bacc_incon_ppmi, std_bacc_incon_ppmi, mean_bacc_con_ppmi, std_bacc_con_ppmi, 'PPMI')

    if mph_flg:
        # MPH dataset (n = 640)
        # Get filenames of CSV tables with sigmoid outputs (one table per random split)
        mph_file_names = [
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_split4.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_split5.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_split6.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_split7.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_split8.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_split9.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_splittest.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_splittrain.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_splittrain0.csv",
            "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority/preds_mph/preds_baseline_majority_randomization_splitvalid.csv",
        ]

        # Do the work for MPH dataset
        bacc_incon_mph = np.zeros((n_splits, n_prop))
        bacc_con_mph = np.zeros((n_splits, n_prop))
        observed_percent_incon_mph = np.zeros((n_splits, n_prop))

        for i in range(n_splits):
            mph_score, mph_true_label = importCSVother(mph_file_names[i])

            mph_score = np.array(mph_score)
            mph_true_label = np.array(mph_true_label)

            mph_score = -mph_score
            observed_percent_incon_mph[i, :], bacc_incon_mph[i, :], bacc_con_mph[i, :] = testInconInterval(mph_score, mph_true_label, cutoff[i], percent_incon, lower_bound[[i], :], upper_bound[[i], :])

        # Average over random splits for MPH dataset
        mean_observed_percent_incon_mph = np.mean(observed_percent_incon_mph, axis=0)
        std_observed_percent_incon_mph = np.std(observed_percent_incon_mph, axis=0)
        mean_bacc_incon_mph = 100 * np.mean(bacc_incon_mph, axis=0, where=~np.isnan(bacc_incon_mph))
        std_bacc_incon_mph = 100 * np.std(bacc_incon_mph, axis=0, where=~np.isnan(bacc_incon_mph))
        mean_bacc_con_mph = 100 * np.mean(bacc_con_mph, axis=0, where=~np.isnan(bacc_con_mph))
        std_bacc_con_mph = 100 * np.std(bacc_con_mph, axis=0, where=~np.isnan(bacc_con_mph))

        # Scaled area under balanced accuracy in cnclusive cases
        relFmph = plotBacc(percent_incon, ind_percent_incon_max, mean_observed_percent_incon_mph, std_observed_percent_incon_mph, mean_bacc_incon_mph, std_bacc_incon_mph, mean_bacc_con_mph, std_bacc_con_mph, 'MPH')




def importCSVdevelopment(fn):
    # Initialize variables
    sample = []
    majorityVote = []
    prediction = []

    # Open and read the CSV file
    with open(fn, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            id, prediction_value, true_label, split1 = row
            sample.append(split1)

            if true_label in ['1.0', '[1, 1, 1]', '[1, 1, 0]', '[1, 0, 1]', '[0, 1, 1]']:
                majorityVote.append(1)
            else:
                majorityVote.append(0)

            prediction.append(float(prediction_value))

    return sample, majorityVote, prediction


def importCSVother(fn):
    # Initialize variables
    prediction = []
    true_label = []

    # Open and read the CSV file
    with open(fn, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            id, prediction_value, true_label_value = row
            prediction.append(float(prediction_value))
            true_label.append(float(true_label_value))

    return prediction, true_label


def ROC(score, trueLabel, doPlot=1):
    # Calculate ROC metrics
    if doPlot != 0:
        print("Plotting ROC is not implemented in this Python conversion.")

    class1 = [i for i, label in enumerate(trueLabel) if label == 1]
    class0 = [i for i, label in enumerate(trueLabel) if label == 0]

    thresh = sorted(list(set(score)))
    Nthresh = len(thresh)
    sens = [0] * Nthresh
    spec = [0] * Nthresh

    for thi in range(Nthresh):
        th = thresh[thi]
        TP = sum(score[i] <= th for i in class1)
        TN = sum(score[i] > th for i in class0)
        sens[thi] = TP / len(class1)
        spec[thi] = TN / len(class0)

    spec = np.array(spec)
    sens = np.array(sens)
    thresh = np.array(thresh)

    AUC = sum(abs(spec[:-1] - spec[1:]) * sens[1:])

    Youden = sens + spec - 1
    maxJind = np.argmax(Youden)
    maxJ = Youden[maxJind]

    cutoff = thresh[maxJind]

    return AUC, cutoff


def crossTable(cutoff, score, trueLabel):
    class1 = [i for i, label in enumerate(trueLabel) if label == 1]
    class0 = [i for i, label in enumerate(trueLabel) if label == 0]

    TP = sum(score[i] < cutoff for i in class1)
    FP = sum(score[i] < cutoff for i in class0)
    TN = sum(score[i] >= cutoff for i in class0)
    FN = sum(score[i] >= cutoff for i in class1)

    return TP, FP, TN, FN


def accMetrics(TP, FP, TN, FN):
    acc = np.divide((TP + TN) , (TP + FP + TN + FN))
    sens = np.divide(TP, (TP + FN))
    spec = np.divide(TN, (FP + TN))
    bacc = 0.5 * (sens + spec)
    PPV = np.divide(TP, (TP + FP))
    NPV = np.divide(TN, (TN + FN))

    return bacc, acc, sens, spec, PPV, NPV


def inconInterval(score, cutoff, percent_incon):
    # Sort the score array
    score = np.sort(score)
    
    # Find the index of the cutoff
    ind_cutoff = np.argmin(np.abs(score - cutoff))
    
    n = len(score)
    lower_bound = []
    upper_bound = []
    
    for p in percent_incon:
        nin = round(p / 100 * n)
        k = round((nin - 1) / 2)
        lower_bound.append(score[ind_cutoff - k])
        upper_bound.append(score[ind_cutoff + k])
    
    return lower_bound, upper_bound



def testInconInterval(score, trueLabel, cutoff, percentIncon, lowerBound, upperBound):
    nSplits = lowerBound.shape[0]
    nProp = len(percentIncon)
    observedPercentIncon = np.zeros((nSplits, nProp))
    baccIncon = np.zeros((nSplits, nProp))  # overall accuracy in inconclusive cases
    baccCon = np.zeros((nSplits, nProp))  # overall accuracy in conclusive cases

    if (nSplits > 1) and (isinstance(cutoff, (int, float, np.float64, np.int64))):
        cutoff = np.array([cutoff] * nSplits)
    else:
        # since cutoff is accessed via index (matlab erlaubts..)
        cutoff = [cutoff]

    for i in range(nSplits):
        for j in range(nProp):
            m = ( (score >= lowerBound[i,j]) & (score <= upperBound[i,j]) )

            observedPercentIncon[i, j] = 100 * np.sum(m) / len(m)

            scoreIncon = score[m]
            trueLabelIncon = trueLabel[m]
            
            TP, FP, TN, FN = crossTable(cutoff[i], scoreIncon, trueLabelIncon)
            baccIncon[i, j] = accMetrics(TP, FP, TN, FN)[0]

            scoreCon = score[~m]
            trueLabelCon = trueLabel[~m]
            TP, FP, TN, FN = crossTable(cutoff[i], scoreCon, trueLabelCon)
            baccCon[i, j] = accMetrics(TP, FP, TN, FN)[0]

    return observedPercentIncon, baccIncon, baccCon



def plotBacc(x, n, obx, dobx, y, dy, z, dz, setID):
    # Calculate the area under the curve scaled to maximum area
    
    obxc, zc = cleanX(obx, z)

    # Perform cubic spline interpolation
    spline = interp1d(x=obxc, y=zc, kind='cubic')

    x_clipped = np.clip(x, min(obxc), max(obxc)) 
    iz = spline(x_clipped[:n])

    # Calculate the relative area under the curve scaled to maximum area
    relF = 100 * np.trapz(iz, x=x[:n]) / (100 * (x[n - 1] - x[0]))

    # Plot observed proportion of inconclusive cases in the test set
    plt.figure()
    plt.errorbar(x[:n], obx[:n], dobx[:n], fmt='b*', linestyle='None', label='observed')
    plt.plot(x[:n], x[:n], '-k', label='identity line')
    plt.legend()
    plt.xlabel('inconclusive cases in validation set (%)')
    plt.ylabel('observed inconclusive cases in the test set (%, mean+/-SD)')
    plt.title(f'{setID} dataset')

    # Plot balanced accuracy in conclusive and inconclusive cases
    plt.figure()
    no = np.argmin(np.abs(obx - x[n - 1]))
    plt.errorbar(obx[:no], y[:no], dy[:no], fmt='ro', linestyle='None', label='inconclusive')
    plt.errorbar(obx[:no], z[:no], dz[:no], fmt='b*', linestyle='None', label='conclusive')
    plt.legend()
    plt.xlabel('mean observed inconclusive cases in the test set (%)')
    plt.ylabel('balanced accuracy (%)')
    plt.title(f'{setID} dataset')
    plt.axis([0, x[n - 1], 0, 100])

    # Plot balanced accuracy in conclusive cases
    plt.figure()
    plt.errorbar(obx[:no], z[:no], dz[:no], fmt='b*', linestyle='None')
    plt.xlabel('mean observed inconclusive cases in the test set (%)')
    plt.ylabel('balanced accuracy (%)')
    plt.title(f'{setID} dataset: relF = {relF:.1f}')
    plt.axis([0, x[n - 1], 90, 100])

    plt.show()


def cleanX(x, y):
    i = 0
    xc = [x[i]]
    yc = [y[i]]
    
    for j in range(1, len(x)):
        if x[j] != xc[i]:
            xc.append(x[j])
            yc.append(y[j])
            i += 1

    return xc, yc



if __name__ == '__main__':
    uncertainty_sigmoid(n_splits=10, target_balanced_accuracy=98)