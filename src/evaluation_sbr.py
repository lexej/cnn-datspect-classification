import os
import shutil
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def uncertaintySBR(methodID, path_to_results_dir: str, nSplits=1, targetBalancedAccuracy=98.0):

    nASC = 2
    nFWHM = 6
    nID = 1740
    sameID = np.arange(1, nASC * nFWHM) * nID

    # Load development data
    ID, ASC, aFWHM, SBRHVputMin, Rmajority = getDevelopment()

    # Load predefined random splits
    ID2, splits = get_randomSplits(nSplits)

    # (Incomplete) Check ordering of SBR and random splits
    if np.sum(np.abs(ID[:len(ID2)] - ID2)) != 0:
        raise ValueError('ERROR: ordering?')

    # Preparations accuracy
    AUC = np.zeros(nSplits)
    cutoff = np.zeros(nSplits)
    bacc = np.zeros((nSplits, 3))
    acc = np.zeros((nSplits, 3))
    sens = np.zeros((nSplits, 3))
    spec = np.zeros((nSplits, 3))
    PPV = np.zeros((nSplits, 3))
    NPV = np.zeros((nSplits, 3))

    # Preparations inconclusive cases
    percentIncon = np.arange(0.2, 50.2, 0.2)
    percentInconMax = 20
    indPercentInconMax = np.argmin(np.abs(percentIncon - percentInconMax)) + 1
    nProp = len(percentIncon)
    lowerBound = np.zeros((nSplits, nProp))
    upperBound = np.zeros((nSplits, nProp))
    observedPercentIncon = np.zeros((nSplits, nProp))
    baccIncon = np.zeros((nSplits, nProp))
    baccCon = np.zeros((nSplits, nProp))

    # Loop over random splits
    for i in range(nSplits):
        # Random split
        splt = splits[:, i]
        indTrain = np.where(splt == 1)[0]
        indValid = np.where(splt == 2)[0]
        indTest = np.where(splt == 3)[0]
        nTrain = len(indTrain)
        nValid = len(indValid)
        nTest = len(indTest)

        # Expand indices to include all images of the same subject
        for j in range(len(sameID)):
            indTrain = np.concatenate([indTrain, indTrain[:nTrain] + sameID[j]])
            indValid = np.concatenate([indValid, indValid[:nValid] + sameID[j]])
            indTest = np.concatenate([indTest, indTest[:nTest] + sameID[j]])

        # Training (fix cutoff for the current random split)
        score = SBRHVputMin[indTrain]
        trueLabel = Rmajority[indTrain]
        AUC[i], cutoff[i] = ROC(score, trueLabel, False)
        TP, FP, TN, FN = crossTable(cutoff[i], score, trueLabel)
        bacc[i, 0], acc[i, 0], sens[i, 0], spec[i, 0], PPV[i, 0], NPV[i, 0] = accMetrics(TP, FP, TN, FN)

        # Validation (fix inconclusive interval)
        score = SBRHVputMin[indValid]
        trueLabel = Rmajority[indValid]
        TP, FP, TN, FN = crossTable(cutoff[i], score, trueLabel)
        bacc[i, 1], acc[i, 1], sens[i, 1], spec[i, 1], PPV[i, 1], NPV[i, 1] = accMetrics(TP, FP, TN, FN)
        lowerBound[i, :], upperBound[i, :] = inconInterval(score, cutoff[i], percentIncon)

        # Testing (accuracy in whole test set)
        score = SBRHVputMin[indTest]
        trueLabel = Rmajority[indTest]
        TP, FP, TN, FN = crossTable(cutoff[i], score, trueLabel)
        bacc[i, 2], acc[i, 2], sens[i, 2], spec[i, 2], PPV[i, 2], NPV[i, 2] = accMetrics(TP, FP, TN, FN)

        # Test inconclusive range in the test set
        score = SBRHVputMin[indTest]
        trueLabel = Rmajority[indTest]
        observedPercentIncon[i, :], baccIncon[i, :], baccCon[i, :] = testInconInterval(score, trueLabel, cutoff[i], percentIncon, lowerBound[[i], :], upperBound[[i], :])

    # Average over random splits
    meanCutoff = np.mean(cutoff)
    stdCutoff = np.std(cutoff)
    meanLowerBound = np.mean(lowerBound, axis=0)
    meanUpperBound = np.mean(upperBound, axis=0)
    stdLowerBound = np.std(lowerBound, axis=0)
    stdUpperBound = np.std(upperBound, axis=0)
    meanObservedPercentIncon = np.mean(observedPercentIncon, axis=0)
    stdObservedPercentIncon = np.std(observedPercentIncon, axis=0)
    meanBaccIncon = 100 * np.nanmean(baccIncon, axis=0)
    stdBaccIncon = 100 * np.nanstd(baccIncon, axis=0)
    meanBaccCon = 100 * np.nanmean(baccCon, axis=0)
    stdBaccCon = 100 * np.nanstd(baccCon, axis=0)

    # Plot lower and upper bound of inconclusive range
    fig, ax = plt.subplots()

    ax.errorbar(percentIncon[:indPercentInconMax], meanLowerBound[:indPercentInconMax],
                yerr=stdLowerBound[:indPercentInconMax], fmt='ro', label='lower bound')

    ax.errorbar(percentIncon[:indPercentInconMax], meanUpperBound[:indPercentInconMax],
                yerr=stdUpperBound[:indPercentInconMax], fmt='b*', label='upper bound')

    ax.errorbar(percentIncon[:indPercentInconMax], meanCutoff * np.ones(indPercentInconMax),
                yerr=stdCutoff * np.ones(indPercentInconMax), fmt='k+', label='cutoff')

    ax.legend()
    ax.set_xlabel('percent inconclusive cases (%)')
    ax.set_ylabel('SBR')

    setID = 'development'

    fig.savefig(os.path.join(path_to_results_dir, f"sbr_percInconclCases_{setID}.png"), dpi=300)

    #plt.show()


    # Primary quality metric (relF)
    relF = plotBacc(percentIncon, indPercentInconMax, meanObservedPercentIncon,
                    stdObservedPercentIncon, meanBaccIncon, stdBaccIncon, meanBaccCon, stdBaccCon, 
                    setID, 
                    methodID,
                    path_to_results_dir)

    # Get proportion of inconclusives at target balanced accuracy in conclusive cases
    ind = np.where(meanBaccCon >= targetBalancedAccuracy)[0]
    percentInconAtTarget = percentIncon[ind[0]]
    meanLowerBoundAtTarget = meanLowerBound[ind[0]]
    meanUpperBoundAtTarget = meanUpperBound[ind[0]]
    stdLowerBoundAtTarget = stdLowerBound[ind[0]]
    stdUpperBoundAtTarget = stdUpperBound[ind[0]]

    # Display results
    performance_dev = (f'\n\nResults:'
    f'\n\tAUC = {np.mean(AUC):.3f}+/-{np.std(AUC):.3f}'
    f'\n\tcutoff = {meanCutoff:.3f}+/-{stdCutoff:.3f}'
    f'\n\tbACC: train = {np.mean(bacc[:, 0]):.3f}+/-{np.std(bacc[:, 0]):.3f} \t'
                f'valid = {np.mean(bacc[:, 1]):.3f}+/-{np.std(bacc[:, 1]):.3f} \t'
                f'test = {np.mean(bacc[:, 2]):.3f}+/-{np.std(bacc[:, 2]):.3f}'
    f'\n\tACC:  train = {np.mean(acc[:, 0]):.3f}+/-{np.std(acc[:, 0]):.3f} \t'
                f'valid = {np.mean(acc[:, 1]):.3f}+/-{np.std(acc[:, 1]):.3f} \t'
                f'test = {np.mean(acc[:, 2]):.3f}+/-{np.std(acc[:, 2]):.3f}'
    f'\n\tSENS: train = {np.mean(sens[:, 0]):.3f}+/-{np.std(sens[:, 0]):.3f} \t'
                f'valid = {np.mean(sens[:, 1]):.3f}+/-{np.std(sens[:, 1]):.3f} \t'
                f'test = {np.mean(sens[:, 2]):.3f}+/-{np.std(sens[:, 2]):.3f}'
    f'\n\tSPEC: train = {np.mean(spec[:, 0]):.3f}+/-{np.std(spec[:, 0]):.3f} \t'
                f'valid = {np.mean(spec[:, 1]):.3f}+/-{np.std(spec[:, 1]):.3f} \t'
                f'test = {np.mean(spec[:, 2]):.3f}+/-{np.std(spec[:, 2]):.3f}'
    f'\n\tPPV:  train = {np.mean(PPV[:, 0]):.3f}+/-{np.std(PPV[:, 0]):.3f} \t'
                f'valid = {np.mean(PPV[:, 1]):.3f}+/-{np.std(PPV[:, 1]):.3f} \t'
                f'test = {np.mean(PPV[:, 2]):.3f}+/-{np.std(PPV[:, 2]):.3f}'
    f'\n\tNPV:  train = {np.mean(NPV[:, 0]):.3f}+/-{np.std(NPV[:, 0]):.3f} \t'
                f'valid = {np.mean(NPV[:, 1]):.3f}+/-{np.std(NPV[:, 1]):.3f} \t'
                f'test = {np.mean(NPV[:, 2]):.3f}+/-{np.std(NPV[:, 2]):.3f}'
    f'\n\tinconclusives at {targetBalancedAccuracy:.1f}% balanced accuracy: {percentInconAtTarget:.1f}%'
    f'\n\tinconclusive SBR range:  {meanLowerBoundAtTarget:.3f}+/-{stdLowerBoundAtTarget:.3f} - '
          f'{meanUpperBoundAtTarget:.3f}+/-{stdUpperBoundAtTarget:.3f}')

    print(performance_dev)

    with open(os.path.join(path_to_results_dir, f"performance_{methodID}_{setID}.txt"), "w") as f:
        f.write(performance_dev)

    #   --------------------------------------------------------------------------------------------------------
    # PPMI dataset
    PPMIscore, PPMItrueLabel = getPPMI()

    # Estimate best possible classification accuracy with a cutoff derived in this dataset
    AUCppmi, cutoffPPMI = ROC(PPMIscore, PPMItrueLabel, False)
    TP, FP, TN, FN = crossTable(cutoffPPMI, PPMIscore, PPMItrueLabel)
    baccPPMI = accMetrics(TP, FP, TN, FN)

    # Perform the work for PPMI dataset
    observedPercentInconPPMI, baccInconPPMI, baccConPPMI = testInconInterval(PPMIscore, PPMItrueLabel, cutoff, percentIncon, lowerBound, upperBound)

    # Average over random splits
    meanObservedPercentInconPPMI = np.mean(observedPercentInconPPMI, axis=0)
    stdObservedPercentInconPPMI = np.std(observedPercentInconPPMI, axis=0)
    meanBaccInconPPMI = 100 * np.nanmean(baccInconPPMI, axis=0)
    stdBaccInconPPMI = 100 * np.nanstd(baccInconPPMI, axis=0)
    meanBaccConPPMI = 100 * np.nanmean(baccConPPMI, axis=0)
    stdBaccConPPMI = 100 * np.nanstd(baccConPPMI, axis=0)

    # Scaled area under balanced accuracy in conclusive cases
    relFppmi = plotBacc(
        percentIncon, indPercentInconMax, meanObservedPercentInconPPMI, stdObservedPercentInconPPMI,
        meanBaccInconPPMI, stdBaccInconPPMI, meanBaccConPPMI, stdBaccConPPMI, 'PPMI', methodID, path_to_results_dir
    )

    #   --------------------------------------------------------------------------------------------------------
    # MPH dataset
    MPHscore, MPHtrueLabel = getMPH()

    # Estimate best possible classification accuracy with a cutoff derived in this dataset
    AUCmph, cutoffMPH = ROC(MPHscore, MPHtrueLabel, False)
    TP, FP, TN, FN = crossTable(cutoffMPH, MPHscore, MPHtrueLabel)
    baccMPH = accMetrics(TP, FP, TN, FN)

    # Perform the work for MPH dataset
    observedPercentInconMPH, baccInconMPH, baccConMPH = testInconInterval(MPHscore, MPHtrueLabel, cutoff, percentIncon, lowerBound, upperBound)

    # Average over random splits
    meanObservedPercentInconMPH = np.mean(observedPercentInconMPH, axis=0)
    stdObservedPercentInconMPH = np.std(observedPercentInconMPH, axis=0)
    meanBaccInconMPH = 100 * np.nanmean(baccInconMPH, axis=0)
    stdBaccInconMPH = 100 * np.nanstd(baccInconMPH, axis=0)
    meanBaccConMPH = 100 * np.nanmean(baccConMPH, axis=0)
    stdBaccConMPH = 100 * np.nanstd(baccConMPH, axis=0)

    # Scaled area under balanced accuracy in conclusive cases
    relFmph = plotBacc(
        percentIncon, indPercentInconMax, meanObservedPercentInconMPH, stdObservedPercentInconMPH,
        meanBaccInconMPH, stdBaccInconMPH, meanBaccConMPH, stdBaccConMPH, 'MPH', methodID, path_to_results_dir
    )


def getDevelopment():
    #   Path to excel file
    excel_file = "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/excel_tabellen/with_smoothing_ROI_analysis_20230725.xlsx"

    df = pd.read_excel(excel_file, sheet_name='results_fpcit_ROI_plus_analysis')

    ID = df['ID'].to_numpy()
    ASC = df['ASC'].to_numpy()
    aFWHM = df['aFWHM'].to_numpy()
    SBRHVputMin = df['SBRHVputMin'].to_numpy()
    Rmajority = df['Rmajority'].to_numpy()

    return ID, ASC, aFWHM, SBRHVputMin, Rmajority


def getPPMI():
    # Path to excel of PPMI
    file_path = "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/PPMI_dataset/PPMI645.xlsx"
    sheet_name = 'Sheet1'

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    PPMIscore = df["SBRputMin"].to_numpy()
    PPMItrueLabel = df["trueLabel"].to_numpy()

    return PPMIscore, PPMItrueLabel


def getMPH():
    # Path to MPH excel
    file_path = "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/MPH_30min_20230824/MPH_30min_20230824.xlsx"
    sheet_name = 'DAT_SPECT_MPH_duration_normaliz'

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    MPHscore = df["SBRHVputMin"].to_numpy()
    MPHtrueLabel = df["scoreConsensus"].to_numpy()

    return MPHscore, MPHtrueLabel


def get_randomSplits(n):
    # Path to excel file
    
    file_path = "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/excel_tabellen/randomSplits03-Aug-2023-17-47-32.xlsx"
    sheet_name = 'splits'

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Replacing non-numeric cells with NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    ID = df["ID"].to_numpy()
    splits = df.iloc[:, 1:n+1].to_numpy()

    return ID, splits


def ROC(score, trueLabel, doPlot=False):
    # Calculate ROC metrics

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

    if doPlot:
        h = plt.figure(figsize=(10.75, 4.2))
    
        localpth = os.path.dirname(os.path.realpath(__file__))
        fname = os.path.join(localpth, 'ROC', datetime.now().strftime('%Y-%m-%d-%H-%M-%S').replace(' ', '-').replace(':', '-'))
        
        plt.subplot(1, 2, 1)
        plt.plot(1 - spec, sens, '-')
        e = 0.05
        plt.axis([0 - e, 1 + e, 0 - e, 1 + e])
        plt.xlabel('1 - specificity')
        plt.ylabel('sensitivity')
        plt.grid(True)
        plt.title(f'AUC={AUC:.3f}')
        
        plt.subplot(1, 2, 2)
        bacc = (sens + spec) / 2
        plt.plot(thresh, bacc, '-')
        plt.xlabel('SBR')
        plt.ylabel('bacc')
        plt.grid(True)
        plt.title(f'cutoff={cutoff:.3f}')

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

    if isinstance(cutoff, (int, float, np.int64, np.float64)):
        # cutoff is scalar -> convert to list
        cutoff = cutoff * np.ones(nSplits)

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


def plotBacc(x, n, obx, dobx, y, dy, z, dz, setID, methodID, path_to_results_dir):
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

    plt.savefig(os.path.join(path_to_results_dir, 
                             f"obsInconclCases_inconclCasesValid_{methodID}_{setID}.png"), dpi=300)

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

    plt.savefig(os.path.join(path_to_results_dir,
                             f"bacc_obsInconclCases_{methodID}_{setID}.png"), dpi=300)

    # Plot balanced accuracy in conclusive cases
    plt.figure()
    plt.errorbar(obx[:no], z[:no], dz[:no], fmt='b*', linestyle='None')
    plt.xlabel('mean observed inconclusive cases in the test set (%)')
    plt.ylabel('balanced accuracy (%)')
    plt.title(f'{setID} dataset: relF = {relF:.1f}')
    plt.axis([0, x[n - 1], 90, 100])

    plt.savefig(os.path.join(path_to_results_dir, 
                             f"bacc_obsInconclCases_concl_{methodID}_{setID}.png"), dpi=300)

    #plt.show()


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

    path_to_results_dir = "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/evaluations"

    if not os.path.exists(path_to_results_dir):
            os.makedirs(path_to_results_dir)

    methodID = "sbr"

    path_to_evaluations_for_method = os.path.join(path_to_results_dir, methodID)

    if os.path.exists(path_to_evaluations_for_method):
            # delete old results
            shutil.rmtree(path_to_evaluations_for_method)
        
    os.makedirs(path_to_evaluations_for_method)

    uncertaintySBR(methodID=methodID,
                   path_to_results_dir=path_to_evaluations_for_method,
                   nSplits=10, 
                   targetBalancedAccuracy=98.0)
