import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation_sigmoid_rfc import inconInterval, testInconInterval, ROC, crossTable, accMetrics, plotBacc, _get_plot_params


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
    
    setID = 'development'

    def create_plot(target_figsize):

        # Plot lower and upper bound of inconclusive range
        fig, ax = plt.subplots(figsize=target_figsize)

        xticks_stepsize, markersize, elinewidth, mask = _get_plot_params(target_figsize=target_figsize, 
                                                                         n_xelements=len(percentIncon[:indPercentInconMax]))

        ax.errorbar(percentIncon[:indPercentInconMax], meanLowerBound[:indPercentInconMax],
                    yerr=stdLowerBound[:indPercentInconMax], 
                    markersize=markersize,
                    elinewidth=elinewidth,
                    markevery=mask,
                    fmt='ro', 
                    markerfacecolor='none', 
                    label='Lower Bound')

        ax.errorbar(percentIncon[:indPercentInconMax], meanUpperBound[:indPercentInconMax],
                    yerr=stdUpperBound[:indPercentInconMax],
                    markersize=markersize,
                    elinewidth=elinewidth,
                    markevery=mask,
                    fmt='b*', 
                    markerfacecolor='none', 
                    label='Upper Bound')

        ax.errorbar(percentIncon[:indPercentInconMax], meanCutoff * np.ones(indPercentInconMax),
                    yerr=stdCutoff * np.ones(indPercentInconMax),
                    markersize=markersize,
                    elinewidth=elinewidth,
                    markevery=mask,
                    fmt='k+', 
                    label='Cutoff')

        ax.legend(loc='upper left')
        ax.set_xlabel('Percentage of Inconclusive Cases (%)')
        ax.set_ylabel('SBR')
        ax.set_xlim(0, percentIncon[indPercentInconMax])
             
        ax.set_xticks(np.arange(0, percentIncon[indPercentInconMax], xticks_stepsize))

        fig.tight_layout()

        leaf_dir = os.path.join(path_to_results_dir, f"{target_figsize[0]}{target_figsize[1]}")
        
        if not os.path.exists(leaf_dir):
            os.makedirs(leaf_dir)

        fig.savefig(os.path.join(leaf_dir,
                                 f"sbr_percInconclCases_{setID}.png"), dpi=300)

        #plt.show()
    
    # 4:3 ratio figsizes (for different use cases)
    target_figsizes = [(8,6), (4, 3)]
    
    for tfsize in target_figsizes:
         create_plot(tfsize)
    

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
