import os
import shutil
import glob
import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl

TARGET_SIZE = 17

mpl.rcParams['font.size'] = TARGET_SIZE
mpl.rcParams['axes.labelsize'] = TARGET_SIZE
mpl.rcParams['axes.titlesize'] = TARGET_SIZE
mpl.rcParams['xtick.labelsize'] = TARGET_SIZE
mpl.rcParams['ytick.labelsize'] = TARGET_SIZE
mpl.rcParams['legend.fontsize'] = TARGET_SIZE


def uncertainty_sigmoid(dir_preds_dev, dir_preds_ppmi, dir_preds_mph, methodID, path_to_results_dir, 
                        n_splits=1, target_balanced_accuracy=98.0):
    ppmi_flg = True
    mph_flg = True

    cutoff_natural = 0.5  # natural cutoff on sigmoid output
    natural_cutoff_flg = True  # use natural cutoff

    # Development dataset (n = 2*6*1740 = 20880)
    # Get filenames of CSV tables with sigmoid outputs (one table per random split)
    
    file_names = glob.glob(os.path.join(dir_preds_dev, '*.csv'))

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
        # Import the data from CSV files
        sample, majority_vote, prediction = importCSVdevelopment(file_names[i])

        sample = np.array(sample)
        majority_vote = np.array(majority_vote)
        prediction = np.array(prediction)

        # Training
        m = (sample == '1')
        score = -prediction[m]
        true_label = majority_vote[m]

        TP, FP, TN, FN = crossTable(-cutoff_natural, score, true_label)
        bacc[i, 0], acc[i, 0], sens[i, 0], spec[i, 0], ppv[i, 0], npv[i, 0] = accMetrics(TP, FP, TN, FN)

        # Validation
        m = (sample == '2')
        score = -prediction[m]
        true_label = majority_vote[m]

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

        TP, FP, TN, FN = crossTable(cutoff[i], score, true_label)
        bacc[i, 2], acc[i, 2], sens[i, 2], spec[i, 2], ppv[i, 2], npv[i, 2] = accMetrics(TP, FP, TN, FN)

        # Test inconclusive range in the test set
        m = (sample == '3')
        score = -prediction[m]
        true_label = majority_vote[m]

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

    setID = 'development'

    # Plot lower and upper bound of inconclusive range
    def create_plot(target_figsize):
    
        fig, ax = plt.subplots(figsize=target_figsize)

        #   ATTENTION: Here lower bound and upper bound are switched 
        #               since the computation was performed on negated pred values

        xticks_stepsize, markersize, elinewidth, mask = _get_plot_params(target_figsize=target_figsize, 
                                                                         n_xelements=len(percent_incon[:ind_percent_incon_max]))

        ax.errorbar(x=percent_incon[:ind_percent_incon_max], 
                    y=mean_lower_bound[:ind_percent_incon_max], 
                    yerr=std_lower_bound[:ind_percent_incon_max], 
                    label='Upper Bound',
                    markersize=markersize,
                    elinewidth=elinewidth,
                    markevery=mask,
                    fmt='b*', markerfacecolor='none')
        ax.errorbar(x=percent_incon[:ind_percent_incon_max], 
                    y=mean_upper_bound[:ind_percent_incon_max], 
                    yerr=std_upper_bound[:ind_percent_incon_max],
                    label='Lower Bound',
                    markersize=markersize,
                    elinewidth=elinewidth,
                    markevery=mask,
                    fmt='ro', markerfacecolor='none')

        ax.set_xlim(0, percent_incon[ind_percent_incon_max])
        ax.set_ylim(0, 1)

        ax.set_xticks(np.arange(0, percent_incon[ind_percent_incon_max], xticks_stepsize))
        ax.set_yticks(np.arange(0, 1, 0.1))

        if natural_cutoff_flg:
            ax.axhline(y=mean_cutoff, label='Cutoff', color='black', linestyle='-')
        else:
            ax.errorbar(x=percent_incon[:ind_percent_incon_max], 
                        y=[mean_cutoff] * ind_percent_incon_max, 
                        yerr=[std_cutoff] * ind_percent_incon_max, 
                        label='Cutoff',
                        color='black', linestyle='-')

        ax.legend(loc='best')
        ax.set_xlabel('Percentage of Inconclusive Cases (%)')
        ax.set_ylabel('Sigmoid Output')

        fig.tight_layout()

        leaf_dir = os.path.join(path_to_results_dir, f"{target_figsize[0]}{target_figsize[1]}")
        
        if not os.path.exists(leaf_dir):
            os.makedirs(leaf_dir)

        fig.savefig(os.path.join(leaf_dir,
                                 f"sigmoid_percInconclCases_{methodID}_{setID}.png"), 
                    dpi=300)

        #plt.show()

    # 4:3 ratio figsizes (for different use cases)
    target_figsizes = [(3,3), (4, 4), (5, 5), (6, 6)]
    
    for tfsize in target_figsizes:
         create_plot(tfsize)


    # Primary quality metric (relF): area under the curve of balanced accuracy in conclusive cases
    # versus proportion of inconclusive cases (scaled to the maximum possible area)
    plotBacc(x=percent_incon, 
             n=ind_percent_incon_max, 
             obx=mean_observed_percent_incon, 
             dobx=std_observed_percent_incon, 
             y=mean_bacc_incon, 
             dy=std_bacc_incon, 
             z=mean_bacc_con, 
             dz=std_bacc_con, 
             setID=setID, 
             methodID=methodID, 
             path_to_results_dir=path_to_results_dir)

    # Get proportion of inconclusives at target balanced accuracy in conclusive cases
    ind = np.where(mean_bacc_con >= target_balanced_accuracy)[0]
    percent_incon_at_target = percent_incon[ind[0]]
    mean_lower_bound_at_target = mean_lower_bound[ind[0]]
    mean_upper_bound_at_target = mean_upper_bound[ind[0]]
    std_lower_bound_at_target = std_lower_bound[ind[0]]
    std_upper_bound_at_target = std_upper_bound[ind[0]]

    # Display results
    performance_dev = (f'\n\nResults:'
    f'\n\tAUC = {np.mean(AUC):.3f}+/-{np.std(AUC):.3f}'
    f'\n\tcutoff = {mean_cutoff:.3f}+/-{std_cutoff:.3f}'
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
    f'\n\tPPV:  train = {np.mean(ppv[:, 0]):.3f}+/-{np.std(ppv[:, 0]):.3f} \t'
                f'valid = {np.mean(ppv[:, 1]):.3f}+/-{np.std(ppv[:, 1]):.3f} \t'
                f'test = {np.mean(ppv[:, 2]):.3f}+/-{np.std(ppv[:, 2]):.3f}'
    f'\n\tNPV:  train = {np.mean(npv[:, 0]):.3f}+/-{np.std(npv[:, 0]):.3f} \t'
                f'valid = {np.mean(npv[:, 1]):.3f}+/-{np.std(npv[:, 1]):.3f} \t'
                f'test = {np.mean(npv[:, 2]):.3f}+/-{np.std(npv[:, 2]):.3f}'
    f'\n\tinconclusives at {target_balanced_accuracy:.1f}% balanced accuracy: {percent_incon_at_target:.1f}%'
    f'\n\tinconclusive SBR range:  {mean_lower_bound_at_target:.3f}+/-{std_lower_bound_at_target:.3f} - '
          f'{mean_upper_bound_at_target:.3f}+/-{std_upper_bound_at_target:.3f}')

    print(performance_dev)

    with open(os.path.join(path_to_results_dir, f"performance_{methodID}_{setID}.txt"), "w") as f:
        f.write(performance_dev)

    if ppmi_flg:
        # PPMI dataset (n = 645)
        # Get filenames of CSV tables with sigmoid outputs (one table per random split)

        ppmi_file_names = glob.glob(os.path.join(dir_preds_ppmi, '*.csv'))

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
        plotBacc(x=percent_incon, 
                 n=ind_percent_incon_max, 
                 obx=mean_observed_percent_incon_ppmi, 
                 dobx=std_observed_percent_incon_ppmi, 
                 y=mean_bacc_incon_ppmi, 
                 dy=std_bacc_incon_ppmi, 
                 z=mean_bacc_con_ppmi, 
                 dz=std_bacc_con_ppmi,
                 setID='PPMI', 
                 methodID=methodID, 
                 path_to_results_dir=path_to_results_dir)

    if mph_flg:
        # MPH dataset (n = 640)
        # Get filenames of CSV tables with sigmoid outputs (one table per random split)

        mph_file_names = glob.glob(os.path.join(dir_preds_mph, '*.csv'))

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

        # Scaled area under balanced accuracy in conclusive cases
        plotBacc(x=percent_incon, 
                 n=ind_percent_incon_max, 
                 obx=mean_observed_percent_incon_mph, 
                 dobx=std_observed_percent_incon_mph, 
                 y=mean_bacc_incon_mph, 
                 dy=std_bacc_incon_mph, 
                 z=mean_bacc_con_mph, 
                 dz=std_bacc_con_mph, 
                 setID='MPH', 
                 methodID=methodID, 
                 path_to_results_dir=path_to_results_dir)


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
    """Gets the score (negated predictions), cutoff (negative) and percent_incon. 
    Returns lower_bound and upper_bound of same length as percent_incon."""

    # Gives sorted negated preds, so that: reduced cases (e.g. -0.92) smaller normal cases (e.g. -0.01)
    score = np.sort(score)

    # cutoff is negated! (e.g. -0.5)
    
    # Find the index of the score-value nearest to cutoff
    ind_cutoff = np.argmin(np.abs(score - cutoff))
    
    n = len(score)
    lower_bound = []
    upper_bound = []
    
    # For each target percentage of inconclusive cases: determine lower and upper bound of inconclusive interval
    for p in percent_incon:
        # number of cases required within inconclusive interval
        nin = round(p / 100 * n)

        # "Middle" of nin -> Assume nin=7 ???
        k = round((nin - 1) / 2)

        #   lower_bound and upper_bound w.r.t. cutoff (which is negative here)
        
        #   Access lower_bound by indexing score (lower_bound tends to reduced class)
        lower_bound.append(score[ind_cutoff - k])

        #   Access upper_bound by indexing score (upper bound tends to normal class)
        upper_bound.append(score[ind_cutoff + k])
    
    return lower_bound, upper_bound



def testInconInterval(score, true_label, cutoff, percent_incon, lower_bound, upper_bound):
    nSplits = lower_bound.shape[0]
    nProp = len(percent_incon)
    observedPercentIncon = np.zeros((nSplits, nProp))
    baccIncon = np.zeros((nSplits, nProp))  # overall accuracy in inconclusive cases
    baccCon = np.zeros((nSplits, nProp))  # overall accuracy in conclusive cases

    if isinstance(cutoff, (int, float, np.int64, np.float64)):
        # cutoff is scalar -> convert to list
        cutoff = cutoff * np.ones(nSplits)

    for i in range(nSplits):
        # For each percentage of inconclusive cases...
        for j in range(nProp):
            # ...compute binary mask of length=len(score); for each score element: True if inconclusive, else False
            m = ( (score >= lower_bound[i,j]) & (score <= upper_bound[i,j]) )

            #   Calculate percentage of inconclusive cases given current lower_bound and upper_bound
            observedPercentIncon[i, j] = 100 * np.sum(m) / len(m)

            scoreIncon = score[m]
            trueLabelIncon = true_label[m]

            #   Calculate balanced accuracy on inconclusive cases
            
            TP, FP, TN, FN = crossTable(cutoff[i], scoreIncon, trueLabelIncon)
            baccIncon[i, j] = accMetrics(TP, FP, TN, FN)[0]

            scoreCon = score[~m]
            trueLabelCon = true_label[~m]

            #   Calculate balanced accuracy on conclusive cases

            TP, FP, TN, FN = crossTable(cutoff[i], scoreCon, trueLabelCon)
            baccCon[i, j] = accMetrics(TP, FP, TN, FN)[0]

    return observedPercentIncon, baccIncon, baccCon



def plotBacc(x, n, obx, dobx, y, dy, z, dz, setID, methodID, path_to_results_dir):
    # Calculate the area under the curve scaled to maximum area
    
    obxc, zc = cleanX(obx, z)

    # Initialize cubic spline interpolator using cleaned z(obx)
    spline = interp1d(x=obxc, y=zc, kind='cubic', fill_value="extrapolate")

    #   Get interpolated z-values given x 
    iz = spline(x[:n])

    #   TODO -> Recheck boundaries
    # Calculate the relative area under the curve scaled to maximum area
    relF = 100 * np.trapz(y=iz, x=x[:n]) / (100 * (x[n - 1] - x[0]))

    ######################################################################################################

    # 1. Plot observed proportion of inconclusive cases in the test set

    def create_obx_x_plot(target_figsize):

        plt.figure(figsize=target_figsize)

        xticks_stepsize, markersize, elinewidth, mask = _get_plot_params(target_figsize=target_figsize, 
                                                                         n_xelements=len(x[:n]))

        plt.errorbar(x[:n], obx[:n], dobx[:n],
                     markersize=markersize,
                     elinewidth=elinewidth,
                     markevery=mask,
                     fmt='b*', linestyle='None', markerfacecolor='none', label='Observed')

        plt.plot(x[:n], x[:n], '-k', label='Identity Line')

        plt.legend(loc='upper left')
        plt.xlabel('Inconclusive Cases in Validation Set (%)')
        plt.ylabel('Observed Inconclusive Cases\nin Test Set (%, mean ± SD)', multialignment='center')
        plt.title(f'{setID} dataset')

        plt.xlim(0, x[n])

        plt.xticks(np.arange(0, x[n], xticks_stepsize))

        plt.tight_layout()

        leaf_dir = os.path.join(path_to_results_dir, f"{target_figsize[0]}{target_figsize[1]}")
        
        if not os.path.exists(leaf_dir):
            os.makedirs(leaf_dir)

        plt.savefig(os.path.join(leaf_dir,
                                 f"obsInconclCases_inconclCasesValid_{methodID}_{setID}.png"), 
                    dpi=300)

    #   index of element in obx nearest to last (considered) element of x
    no = np.argmin(np.abs(obx - x[n - 1]))

    ######################################################################################################
    
    # 2. Plot balanced accuracy in conclusive and inconclusive cases

    def create_y_z_obx_plot(target_figsize):

        plt.figure(figsize=target_figsize)

        xticks_stepsize, markersize, elinewidth, mask = _get_plot_params(target_figsize=target_figsize, 
                                                                         n_xelements=len(obx[:no]))

        plt.errorbar(obx[:no], y[:no], dy[:no],
                     markersize=markersize,
                     elinewidth=elinewidth,
                     markevery=mask,
                     fmt='ro', linestyle='None', markerfacecolor='none', label='Inconclusive Cases')
        plt.errorbar(obx[:no], z[:no], dz[:no],
                     markersize=markersize,
                     elinewidth=elinewidth,
                     markevery=mask,
                     fmt='b*', linestyle='None', markerfacecolor='none', label='Conclusive Cases')

        plt.legend(loc='lower right')
        plt.xlabel('Inconclusive Cases in Test Set (%, mean)')
        plt.ylabel('Balanced Accuracy (%)')
        plt.title(f'{setID} dataset')

        plt.xlim(0, x[n])
        plt.ylim(0, 100)
        plt.xticks(np.arange(0, x[n], xticks_stepsize))
        plt.yticks(np.arange(0, 110, 10.0))

        plt.tight_layout()

        leaf_dir = os.path.join(path_to_results_dir, f"{target_figsize[0]}{target_figsize[1]}")
        
        if not os.path.exists(leaf_dir):
            os.makedirs(leaf_dir)

        plt.savefig(os.path.join(leaf_dir,
                                 f"bacc_obsInconclCases_{methodID}_{setID}.png"), 
                    dpi=300)

    ######################################################################################################
    
    # 3. Plot balanced accuracy in conclusive cases

    def create_z_obx_plot(target_figsize):

        plt.figure(figsize=target_figsize)

        xticks_stepsize, markersize, elinewidth, mask = _get_plot_params(target_figsize=target_figsize, 
                                                                         n_xelements=len(obx[:no]))

        plt.errorbar(obx[:no], z[:no], dz[:no], 
                     markersize=markersize,
                     elinewidth=elinewidth,
                     markevery=mask,
                     fmt='b*', linestyle='None', markerfacecolor='none')
        
        #   Highlight area under mean bacc
        plt.fill_between(x=obx[:no], y1=z[:no], y2=0,
                         label=f'Relative Area Size: {round(relF, 2)}%', 
                         alpha=0.5, 
                         hatch='x', 
                         color='lightblue')

        plt.xlabel('Inconclusive Cases in Test Set (%, mean)')
        plt.ylabel('Balanced Accuracy\non Conclusive Cases (%)',  multialignment='center')
        plt.title(f'{setID} dataset')
        plt.legend()

        plt.xlim(0, x[n])
        plt.ylim(90, 100)
        plt.xticks(np.arange(0, x[n], xticks_stepsize))
        plt.yticks(np.arange(90, 101, 1.0))

        plt.tight_layout()

        leaf_dir = os.path.join(path_to_results_dir, f"{target_figsize[0]}{target_figsize[1]}")
        
        if not os.path.exists(leaf_dir):
            os.makedirs(leaf_dir)

        plt.savefig(os.path.join(leaf_dir,
                                 f"bacc_obsInconclCases_concl_{methodID}_{setID}.png"), 
                    dpi=300)

        #plt.show()

    target_figsizes = [(3,3), (4, 4), (5, 5), (6, 6)]

    for tfsize in target_figsizes:
        create_obx_x_plot(tfsize)
        create_y_z_obx_plot(tfsize)
        create_z_obx_plot(tfsize)


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


def _get_plot_params(target_figsize, n_xelements):
    #   Defaults (if None -> matplotlib takes default)
    xticks_stepsize = 1.0
    markersize = None
    elinewidth = None
    mask = None
    
    if target_figsize[0] < 8:
        #   Small sized figure
        xticks_stepsize = 2.0

        if n_xelements > 50:
            markersize = 4.0
            elinewidth = 1.0
            #   Marker only every second element for better visualization
            mask = np.arange(n_xelements) % 2 == 0
    
    return xticks_stepsize, markersize, elinewidth, mask


if __name__ == '__main__':
    path_to_results_dir = "/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results"

    path_to_results_evaluations_dir = os.path.join(path_to_results_dir, "evaluations")

    if not os.path.exists(path_to_results_evaluations_dir):
            os.makedirs(path_to_results_evaluations_dir)

    #   Methods (with preds) to calculate the evaluations for (EXCEPT SBR!)
    methods = ["pca_rfc", 
               "baseline_majority", 
               "baseline_random",
               "regression"]
    
    for methodID in methods:
        print('*' * 100)

        print(f"----- Started calculation for method {methodID} ------")

        #   Source paths

        dir_preds_dev = os.path.join(path_to_results_dir, methodID, f"preds_{methodID}")
        dir_preds_ppmi = os.path.join(path_to_results_dir, methodID, "preds_ppmi")
        dir_preds_mph = os.path.join(path_to_results_dir, methodID, "preds_mph")

        #   Target path
        
        path_to_evaluations_for_method = os.path.join(path_to_results_evaluations_dir, methodID)

        if os.path.exists(path_to_evaluations_for_method):
            # delete old results
            shutil.rmtree(path_to_evaluations_for_method)
            
        os.makedirs(path_to_evaluations_for_method)

        #   Calculate results

        uncertainty_sigmoid(dir_preds_dev=dir_preds_dev, 
                            dir_preds_ppmi=dir_preds_ppmi,
                            dir_preds_mph=dir_preds_mph,
                            methodID=methodID,
                            path_to_results_dir=path_to_evaluations_for_method,
                            n_splits=10, 
                            target_balanced_accuracy=98)
        
        print(f"----- Finished calculation for method {methodID} ------")

    