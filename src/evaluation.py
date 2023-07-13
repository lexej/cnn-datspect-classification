import sys

from common import os, json, np, pd, plt, sns
from common import torch, DataLoader
from common import f1_score, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, \
    ConfusionMatrixDisplay


class PerformanceEvaluator:

    def __init__(self, model, model_weights_path: str, best_epoch: int, test_dataloader: DataLoader,
                 strategy, results_path):
        self.model = model
        self.model_weights_path = model_weights_path
        self.best_epoch = best_epoch
        self.test_dataloader = test_dataloader
        self.strategy = strategy
        self.results_testing_path = os.path.join(results_path, 'testing')
        os.makedirs(self.results_testing_path, exist_ok=True)

        self.relevant_digits = 5

    def evaluate_on_test_data(self):

        print(f'\n--- Evaluating model (best epoch: {self.best_epoch}) on test data ---')

        #   ---------------------------------------------------------------------------------------------

        ids, preds, labels_original_list = self.__get_predictions()

        #   labels for testing have to be inferred from original labels using a majority vote

        trues_chosen_majority = np.apply_along_axis(lambda l: np.argmax(np.bincount(l)),
                                                    axis=1,
                                                    arr=labels_original_list)

        #   Calculate indices of consensus cases

        consensus_condition = (labels_original_list == 1).all(axis=1) | (labels_original_list == 0).all(axis=1)

        consensus_indices = np.where(consensus_condition)[0]
        no_consensus_indices = np.where(~consensus_condition)[0]

        #   Calculate ids, trues and preds for consensus and no-consensus test cases

        ids_consensus = ids[consensus_indices]
        ids_no_consensus = ids[no_consensus_indices]

        trues_consensus_full = labels_original_list[consensus_indices]

        trues_consensus_reduced = np.apply_along_axis(lambda l: np.argmax(np.bincount(l)),
                                                      axis=1,
                                                      arr=trues_consensus_full)

        trues_no_consensus_full = labels_original_list[no_consensus_indices]

        preds_consensus = preds[consensus_indices]
        preds_no_consensus = preds[no_consensus_indices]

        #   ---------------------------------------------------------------------------------------------

        #   Save predictions for consensus test cases
        self.__save_preds(ids_consensus, preds_consensus, trues_consensus_full,
                          save_as='preds_consensus_cases.csv')

        #   Save predictions for "no consensus" test cases
        self.__save_preds(ids_no_consensus, preds_no_consensus, trues_no_consensus_full,
                          save_as='preds_no_consensus_cases.csv')

        self.__calculate_statistics(preds_consensus=preds_consensus,
                                    trues_consensus_reduced=trues_consensus_reduced,
                                    preds_no_consensus=preds_no_consensus,
                                    trues_no_consensus_full=trues_no_consensus_full,
                                    save_as='preds_statistics.csv')

        x_plot = np.concatenate((preds_consensus, preds_no_consensus))
        y_plot = np.concatenate((np.full_like(preds_consensus, 'consensus', dtype=np.object_),
                                 np.full_like(preds_no_consensus, 'no consensus', dtype=np.object_)))
        hue_plot = np.concatenate((np.apply_along_axis(lambda a: f'label {a}', axis=1, arr=trues_consensus_full),
                                   np.apply_along_axis(lambda a: f'label {a}', axis=1, arr=trues_no_consensus_full)))

        self.__create_stripplot_for_preds(x=x_plot, y=y_plot, hue=hue_plot, save_as='strip_plot.png')

        self.__create_histplot_for_preds(x=x_plot, y=y_plot, hue=hue_plot, save_as='histogram.png')

        if self.strategy == 'baseline' or self.strategy == 'regression':

            #   ---------------------------------------------------------------------------------------------
            #   ROC curves

            #   ROC curve for test data with label consensus

            self.__create_roc_curve(trues=trues_consensus_reduced,
                                    preds=preds_consensus,
                                    title=f'ROC Curve (only test data with label consensus)',
                                    save_as='roc_curve_consensus_cases.png')

            #   ROC curve for all test data; if no consensus in labels: majority vote

            self.__create_roc_curve(trues=trues_chosen_majority,
                                    preds=preds,
                                    title=f'ROC Curve (all test data; majority label choice if no label consensus)',
                                    save_as='roc_curve_all_cases.png')

            #   ---------------------------------------------------------------------------------------------

            #   Define upper and lower threshold for decision

            neg_pred_threshold = 0.1
            pos_pred_threshold = 0.9

            #   For consensus cases
            self.__compute_performance_for_threshold_range(preds=preds_consensus,
                                                           trues=trues_consensus_full,
                                                           ids=ids_consensus,
                                                           lower_threshold=neg_pred_threshold,
                                                           upper_threshold=pos_pred_threshold)

            #   TODO: for all cases (using majority vote)

        print(f'\n--- Evaluation done. ---')

    def __get_predictions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Load weights into model
        self.model.load_state_dict(torch.load(self.model_weights_path))

        #   Evaluation mode
        self.model.eval()

        preds = []
        ids = []
        labels_original = []

        #   Create numpy array for predictions and ground truths

        with torch.no_grad():
            for batch in self.test_dataloader:
                batch_features, batch_labels, batch_metadata = batch

                outputs = self.model(batch_features)

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

        #   Reduce list of dict to list of list (removing information about rater)
        labels = np.vectorize(lambda d: np.array([d[k] for k in ['R1', 'R2', 'R3']]),
                              signature='()->(m)')(labels_original)

        return ids, preds, labels

    def __save_preds(self, ids: np.ndarray, preds: np.ndarray, trues: np.ndarray, save_as: str):

        #   Store predictions for "consensus" and "no consensus" cases

        result = pd.DataFrame({
            'id': ids.tolist(),
            'prediction': preds.tolist(),
            'true_label': trues.tolist()
        })

        result.to_csv(os.path.join(self.results_testing_path, save_as), index=False)

    def __calculate_statistics(self, preds_consensus, trues_consensus_reduced, preds_no_consensus,
                               trues_no_consensus_full, save_as: str):

        #   Reduction using majority vote here:
        trues_no_consensus_reduced = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                                         axis=1,
                                                         arr=trues_no_consensus_full)

        #   Get preds for true negative and true positive consensus cases

        consensus_true_neg_indices = np.where(trues_consensus_reduced == 0)
        consensus_true_pos_indices = np.where(trues_consensus_reduced == 1)

        preds_consensus_true_neg = preds_consensus[consensus_true_neg_indices]
        preds_consensus_true_pos = preds_consensus[consensus_true_pos_indices]

        #   Get preds for "no consensus" cases where majority is positive/negative

        no_consensus_majority_neg_indices = np.where(trues_no_consensus_reduced == 0)
        no_consensus_majority_pos_indices = np.where(trues_no_consensus_reduced == 1)

        preds_no_consensus_majority_neg = preds_no_consensus[no_consensus_majority_neg_indices]
        preds_no_consensus_majority_pos = preds_no_consensus[no_consensus_majority_pos_indices]

        #   ---------------------------------------------------------------------------------------------
        #   Calculate statistics for consensus cases

        def get_statistics_for_preds(predictions: np.ndarray) -> pd.Series:
            mean = float(np.mean(predictions))
            median = float(np.median(predictions))
            standard_dev = float(np.std(predictions))

            result = {
                'mean': mean,
                'median': median,
                'standard_dev': standard_dev

            }
            result = pd.Series(result)

            return result

        #   Statistics for predictions of different test cases

        statistics_on_test_split = pd.DataFrame(columns=['mean', 'median', 'standard_dev'],
                                                index=['consensus_true_negatives', 'consensus_true_positives',
                                                       'no_consensus_majority_neg', 'no_consensus_majority_pos'])

        statistics_on_test_split.loc['consensus_true_negatives'] = get_statistics_for_preds(preds_consensus_true_neg)
        statistics_on_test_split.loc['consensus_true_positives'] = get_statistics_for_preds(preds_consensus_true_pos)

        statistics_on_test_split.loc['no_consensus_majority_neg'] = get_statistics_for_preds(
            preds_no_consensus_majority_neg)

        statistics_on_test_split.loc['no_consensus_majority_pos'] = get_statistics_for_preds(
            preds_no_consensus_majority_pos)

        statistics_on_test_split.to_csv(os.path.join(self.results_testing_path, save_as))

    def __create_stripplot_for_preds(self, x: np.ndarray, y: np.ndarray, hue: np.ndarray, save_as: str):

        #   Strip Plot with points representing the test samples and x-axis is predicted prob

        plt.figure(figsize=(16, 8))

        sns.stripplot(x=x, y=y, hue=hue)

        plt.xlabel('Predicted Probabilities')
        plt.title('Predictions for certain and uncertain test cases - Evaluation on test data')

        # Vertical threshold lines
        for i in np.arange(0, 1.1, 0.1):
            plt.axvline(x=i, linestyle='dotted', color='grey')

        plt.savefig(os.path.join(self.results_testing_path, save_as), dpi=300)

    def __create_histplot_for_preds(self, x: np.ndarray, y: np.ndarray, hue: np.ndarray, save_as: str):

        #   Histogram over predictions

        fig, ax1 = plt.subplots(figsize=(12, 6))

        num_bins = 50
        log_scale = (False, True)

        hue_histplot = [y[i] + '; ' + hue[i] for i in range(len(hue))]

        sns.histplot(x=x,
                     hue=hue_histplot,
                     bins=num_bins,
                     multiple='stack',
                     log_scale=log_scale)

        ax1.set_ylabel('Frequency')

        ax2 = ax1.twinx()
        ax2.set_ylabel('ECDF')

        sns.ecdfplot(x, ax=ax2, color='red', linestyle='dotted')

        #   Mark 5 % "center" region (w.r.t. percentiles):

        ymax_percentiles = 0.6

        percentile_47_5 = np.percentile(x, 50 - 2.5)  # 47.5% Percentile
        plt.axvline(x=percentile_47_5, color='blue', linestyle='--', ymax=ymax_percentiles)
        plt.text(x=percentile_47_5, y=ymax_percentiles, s=f'47.5th Percentile', ha='center')

        percentile_52_5 = np.percentile(x, 50 + 2.5)  # 52.5% Percentile
        plt.axvline(x=percentile_52_5, color='blue', linestyle='--', ymax=ymax_percentiles)
        plt.text(x=percentile_52_5, y=ymax_percentiles, s=f'52.5th Percentile', ha='center')

        # Vertical threshold lines
        for i in np.arange(0, 1.1, 0.1):
            plt.axvline(x=i, linestyle='dotted', color='grey')

        plt.xlabel('Predicted Probabilities')
        plt.title('Histogram - Evaluation on test data')

        plt.savefig(os.path.join(self.results_testing_path, save_as), dpi=300)

    def __create_roc_curve(self, trues, preds, title, save_as: str):

        fpr, tpr, _ = roc_curve(trues, preds)

        roc_auc = auc(fpr, tpr)

        label = f'ROC curve; AUC = {round(roc_auc, self.relevant_digits)}'

        plt.figure(figsize=(12, 6))
        plt.plot(fpr, tpr, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')

        plt.savefig(os.path.join(self.results_testing_path, save_as), dpi=300)

    def __compute_performance_for_threshold_range(self, preds, trues, ids, lower_threshold, upper_threshold):

        #   ---------------------------------------------------------------------------------------------

        #   Sub-path of testing directory for performance evaluation given thresholds

        target_path = os.path.join(self.results_testing_path, f'thresholds_{lower_threshold}_{upper_threshold}')
        os.makedirs(target_path, exist_ok=True)

        #   ---------------------------------------------------------------------------------------------

        trues_reduced = np.apply_along_axis(lambda l: np.argmax(np.bincount(l)), axis=1, arr=trues)

        #   Inbetween both thresholds -> Inconclusive range

        #   Store predictions for misclassified (fp or fn) cases (given a threshold)

        #   fp consensus cases (inconclusive also counted as false)

        false_positive_indices = np.where((preds > lower_threshold) & (trues_reduced == 0))[0]

        fp_samples = pd.DataFrame({
            'id': ids[false_positive_indices].tolist(),
            'prediction': preds[false_positive_indices].tolist(),
            'true_label': trues[false_positive_indices].tolist(),
            'lower_threshold': lower_threshold
        })

        fp_samples.to_csv(os.path.join(target_path, 'fp_samples_consensus.csv'), index=False)

        #   fn consensus cases (inconclusive also counted as false)

        false_negative_indices = np.where((preds < upper_threshold) & (trues_reduced == 1))[0]

        fn_samples = pd.DataFrame({
            'id': ids[false_negative_indices].tolist(),
            'prediction': preds[false_negative_indices].tolist(),
            'true_label': trues[false_negative_indices].tolist(),
            'upper_threshold': upper_threshold
        })

        fn_samples.to_csv(os.path.join(target_path, 'fn_samples_consensus.csv'), index=False)

        #   ---------------------------------------------------------------------------------------------

        #   Calculate metrics precision, recall, f1-score, conf_matrix for label consensus test cases

        #   3 classes: -1 (inconclusive), 0 (normal), 1 (reduced)
        preds = np.where(preds >= upper_threshold, 1, np.where(preds <= lower_threshold, 0, -1))

        #   Calculate metrics per class
        average = None

        acc_score = accuracy_score(trues_reduced, preds)
        precision = precision_score(trues_reduced, preds, average=average)
        recall = recall_score(trues_reduced, preds, average=average, zero_division=0)
        f1 = f1_score(trues_reduced, preds, average=average)

        results_test_split = {
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'labels': ['inconclusive', 'normal', 'reduced'],
            'accuracy': round(acc_score, self.relevant_digits),
            'precision': [round(num, self.relevant_digits) for num in precision],
            'recall': [round(num, self.relevant_digits) for num in recall],
            'f1-score': [round(num, self.relevant_digits) for num in f1]
        }

        with open(os.path.join(target_path, 'perf_metrics_consensus.json'), 'w') as f:
            json.dump(results_test_split, f, indent=4)

        conf_matrix = confusion_matrix(trues_reduced, preds, labels=[-1, 0, 1])

        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                            display_labels=['inconclusive', '0', '1'])
        fig, ax = plt.subplots(figsize=(12, 8))
        cm_display.plot(ax=ax)
        ax.set_title(f"Confusion Matrix for label consensus test cases \n"
                     f"(lower threshold = {lower_threshold}, upper threshold = {upper_threshold})")
        plt.savefig(os.path.join(target_path, 'conf_matrix_consensus.png'), dpi=300)
