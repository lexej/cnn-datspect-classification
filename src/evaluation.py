from common import os, json, np, pd, plt, sns
from common import torch, DataLoader
from common import f1_score, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, \
    ConfusionMatrixDisplay


def evaluate_on_test_data(model, best_epoch_weights_path: str, best_epoch: int, test_dataloader: DataLoader,
                          strategy, results_path):

    results_testing_path = os.path.join(results_path, 'testing')
    os.makedirs(results_testing_path, exist_ok=True)

    relevant_digits = 5

    # Load weights into model
    model.load_state_dict(torch.load(best_epoch_weights_path))

    #   Evaluation mode
    model.eval()

    preds = []
    ids = []
    labels_original = []

    #   Create numpy array for predictions and ground truths

    with torch.no_grad():
        for batch in test_dataloader:
            batch_features, batch_labels, batch_metadata = batch

            outputs = model(batch_features)

            preds.extend(outputs.tolist())

            #   batch_labels shape: dict with keys
            #   'R1', 'R2', 'R3', 'R1S1', 'R1S2', 'R2S1', 'R2S2', 'R3S1', 'R3S2'
            #   and values are torch tensors containing the <batch_size>-labels

            for i in range(len(batch_labels['R1'])):
                labels_sample = {}

                for key, value in batch_labels.items():
                    labels_sample[key] = int(value[i])

                labels_original.append(labels_sample)

            ids.extend(batch_metadata['id'].tolist())

    preds = np.array(preds).squeeze()
    ids = np.array(ids)
    labels_original = np.array(labels_original)

    labels_original_list = np.vectorize(lambda d: np.array([d[k] for k in ['R1', 'R2', 'R3']]),
                                        signature='()->(m)')(labels_original)

    #   labels for testing have to be inferred from original labels using a majority vote

    trues_chosen_majority = np.apply_along_axis(lambda l: np.argmax(np.bincount(l)),
                                                axis=1,
                                                arr=labels_original_list)

    #   Calculate indices of consensus cases

    consensus_condition = (labels_original_list == 1).all(axis=1) | (labels_original_list == 0).all(axis=1)

    consensus_indices = np.where(consensus_condition)[0]
    no_consensus_indices = np.where(~consensus_condition)[0]

    #   Calculate trues and preds for consensus test cases

    ids_consensus = ids[consensus_indices]

    trues_consensus_full = labels_original_list[consensus_indices]

    trues_consensus_reduced = trues_consensus_full[:, 0:1].squeeze()

    preds_consensus = preds[consensus_indices]

    #   Calculate trues and preds for no consensus test cases

    trues_no_consensus_full = labels_original_list[no_consensus_indices]

    #   Select majority value as label
    trues_no_consensus_reduced = np.apply_along_axis(lambda l: np.argmax(np.bincount(l)),
                                                     axis=1,
                                                     arr=trues_no_consensus_full)

    preds_no_consensus = preds[no_consensus_indices]

    #   For binary classification: Compute multiple metrics

    if strategy == 0:

        #   ---------------------------------------------------------------------------------------------
        #   ROC curves

        def create_roc_curve(fpr, tpr, title: str, label: str, file_name_save: str):
            plt.figure(figsize=(12, 6))
            plt.plot(fpr, tpr, label=label)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc='lower right')

            plt.savefig(os.path.join(results_testing_path, file_name_save+'.png'), dpi=300)

        #   ROC curve for test data with label consensus

        fpr, tpr, _ = roc_curve(trues_consensus_reduced, preds_consensus)

        roc_auc_consensus_labels = auc(fpr, tpr)

        create_roc_curve(fpr=fpr,
                         tpr=tpr,
                         title=f'ROC Curve (only test data with label consensus)',
                         label=f'ROC curve; AUC = {round(roc_auc_consensus_labels, relevant_digits)}',
                         file_name_save='roc_curve_consensus_labels')

        #   ROC curve for all test data; if no consensus in labels: majority vote

        fpr, tpr, _ = roc_curve(trues_chosen_majority, preds)

        roc_auc_all_labels = auc(fpr, tpr)

        create_roc_curve(fpr=fpr,
                         tpr=tpr,
                         title=f'ROC Curve (all test data; if no label consensus: majority)',
                         label=f'ROC curve; AUC = {round(roc_auc_all_labels, relevant_digits)}',
                         file_name_save='roc_curve_all_labels')

        #   ---------------------------------------------------------------------------------------------
        #   Strip Plot with points representing the test samples and x-axis is predicted prob

        plt.figure(figsize=(16, 8))

        x_stripplot = np.concatenate((preds_consensus, preds_no_consensus))
        y_stripplot = np.concatenate((np.full_like(preds_consensus, 'consensus', dtype=np.object_),
                                      np.full_like(preds_no_consensus, 'majority', dtype=np.object_)))
        hue_stripplot = np.concatenate((np.apply_along_axis(lambda a: f'label {a}',
                                                            axis=1, arr=trues_consensus_full),
                                        np.apply_along_axis(lambda a: f'label {a}',
                                                            axis=1, arr=trues_no_consensus_full)))

        sns.stripplot(x=x_stripplot, y=y_stripplot, hue=hue_stripplot)

        plt.xlabel('Predicted Probabilities')
        plt.title('Predictions for certain and uncertain test cases - Evaluation on test data')

        # Vertical threshold lines
        for i in np.arange(0, 1.1, 0.1):
            plt.axvline(x=i, linestyle='dotted', color='grey')

        plt.savefig(os.path.join(results_testing_path, 'strip_plot.png'), dpi=300)

        #   ---------------------------------------------------------------------------------------------
        #   Histogram

        plt.figure(figsize=(12, 6))

        num_bins = 50
        log_scale = (False, True)

        x_histplot = x_stripplot
        hue_histplot = [y_stripplot[i] + '; ' + hue_stripplot[i] for i in range(len(hue_stripplot))]

        sns.histplot(x=x_histplot,
                     hue=hue_histplot,
                     bins=num_bins,
                     multiple='stack',
                     log_scale=log_scale)

        # Vertical threshold lines
        for i in np.arange(0, 1.1, 0.1):
            plt.axvline(x=i, linestyle='dotted', color='grey')

        plt.xlabel('Predicted Probabilities')
        plt.ylabel('Frequency')
        plt.title('Histogram - Evaluation on test data')

        plt.savefig(os.path.join(results_testing_path, 'histogram.png'), dpi=300)

        #   ---------------------------------------------------------------------------------------------

        #   Get information about ID's and original labels of misclassified cases (given thresholds)

        #   False positives and false negatives are calculated only for label consensus test cases
        #   (not for uncertain test cases)

        #   fp consensus test cases

        #   TODO: Find optimum !!!
        fp_threshold = 0.1

        false_positive_indices = np.where((preds_consensus > fp_threshold) & (trues_consensus_reduced == 0))[0]

        fp_samples_dict = {
            'id': ids_consensus[false_positive_indices].tolist(),
            'prediction': preds_consensus[false_positive_indices].tolist(),
            'labels_original': trues_consensus_full[false_positive_indices].tolist(),
            'fp_threshold': fp_threshold
        }

        fp_samples = pd.DataFrame(fp_samples_dict)
        fp_samples.to_csv(os.path.join(results_testing_path, 'fp_samples_consensus.csv'), index=False)

        #   fn consensus test cases

        #   TODO: Find optimum !!!
        fn_threshold = 0.9

        false_negative_indices = np.where((preds_consensus < fn_threshold) & (trues_consensus_reduced == 1))[0]

        fn_samples_dict = {
            'id': ids_consensus[false_negative_indices].tolist(),
            'prediction': preds_consensus[false_negative_indices].tolist(),
            'labels_original': trues_consensus_full[false_negative_indices].tolist(),
            'fn_threshold': fn_threshold
        }

        fn_samples = pd.DataFrame(fn_samples_dict)
        fn_samples.to_csv(os.path.join(results_testing_path, 'fn_samples_consensus.csv'), index=False)

        #   ---------------------------------------------------------------------------------------------

        #   TODO: Calculate the Fischer Linear Discriminant (FLD) from preds and trues_chosen_majority

    #   TODO: choose threshold_value appropriately
    threshold_value = 0.5

    #   Calculate metrics acc_score, precision, recall, f1-score, conf_matrix for label consensus test cases!

    if strategy == 0:
        preds_consensus = (preds_consensus > threshold_value).astype(float)
        average = 'binary'

    #   TODO -> Anpassen an strategy 1 und 2
    """
    elif strategy == 1:
        #   Decode predictions by taking argmax (vanilla multi-class..)
        preds = np.argmax(preds, axis=1)
        average = 'macro'
    elif strategy == 2:
        #   TODO: convert torch tensor expectations for numpy

        #   1. Manually decode labels
        trues = enc.decode_labels(trues)

        #   2. Manually decode predictions
        preds = enc.decode_preds(preds)
        
        average = 'macro'
    """

    acc_score = accuracy_score(trues_consensus_reduced, preds_consensus)
    precision = precision_score(trues_consensus_reduced, preds_consensus, average=average)
    recall = recall_score(trues_consensus_reduced, preds_consensus, average=average)
    f1 = f1_score(trues_consensus_reduced, preds_consensus, average=average)

    conf_matrix = confusion_matrix(trues_consensus_reduced, preds_consensus)

    print(f'\nEvaluation of model (best epoch: {best_epoch}) on test split:\n')

    print(f"Accuracy: {round(acc_score, relevant_digits)}")
    print(f"Precision: {round(precision, relevant_digits)}")
    print(f"Recall: {round(recall, relevant_digits)}")
    print(f"F1-score: {round(f1, relevant_digits)}")
    print('\nConfusion matrix:')
    print(conf_matrix)

    #   Save performance metrics

    results_test_split = {
        'threshold_chosen': threshold_value,
        'accuracy': acc_score,
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }

    with open(os.path.join(results_testing_path, 'perf_metrics_consensus.json'), 'w') as f:
        json.dump(results_test_split, f, indent=4)

    cm_display = ConfusionMatrixDisplay(conf_matrix)
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(ax=ax)
    ax.set_title(f"Confusion Matrix for label consensus test cases (threshold = {threshold_value})")
    plt.savefig(os.path.join(results_testing_path, 'conf_matrix_consensus.png'), dpi=300)
