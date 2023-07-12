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

    def evaluate_on_test_data(self):

        relevant_digits = 5

        ids, preds, labels_original_list = self.__get_predictions()

        #   labels for testing have to be inferred from original labels using a majority vote

        trues_chosen_majority = np.apply_along_axis(lambda l: np.argmax(np.bincount(l)),
                                                    axis=1,
                                                    arr=labels_original_list)

        #   Calculate indices of consensus cases

        consensus_condition = (labels_original_list == 1).all(axis=1) | (labels_original_list == 0).all(axis=1)

        consensus_indices = np.where(consensus_condition)[0]
        no_consensus_indices = np.where(~consensus_condition)[0]

        #   ---------------------------------------------------------------------------------------------
        #   Calculate ids, trues and preds for consensus and no-consensus test cases

        ids_consensus = ids[consensus_indices]
        ids_no_consensus = ids[no_consensus_indices]

        trues_consensus_full = labels_original_list[consensus_indices]
        trues_consensus_reduced = trues_consensus_full[:, 0:1].squeeze()
        trues_no_consensus_full = labels_original_list[no_consensus_indices]
        #   Reduction using majority vote here:
        trues_no_consensus_reduced = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                                         axis=1,
                                                         arr=trues_no_consensus_full)

        preds_consensus = preds[consensus_indices]
        preds_no_consensus = preds[no_consensus_indices]

        print('---------------------------------------------------------------------------------------------')
        print(f'\nEvaluation of model (best epoch: {self.best_epoch}) on test split:\n')

        self.__calculate_statistics(preds_consensus=preds_consensus,
                                    trues_consensus_reduced=trues_consensus_reduced,
                                    preds_no_consensus=preds_no_consensus,
                                    trues_no_consensus_reduced=trues_no_consensus_reduced)

        #   ---------------------------------------------------------------------------------------------
        #   Strip Plot with points representing the test samples and x-axis is predicted prob

        plt.figure(figsize=(16, 8))

        x_stripplot = np.concatenate((preds_consensus, preds_no_consensus))
        y_stripplot = np.concatenate((np.full_like(preds_consensus, 'consensus', dtype=np.object_),
                                      np.full_like(preds_no_consensus, 'no consensus', dtype=np.object_)))
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

        plt.savefig(os.path.join(self.results_testing_path, 'strip_plot.png'), dpi=300)

        #   ---------------------------------------------------------------------------------------------
        #   Histogram over predictions

        fig, ax1 = plt.subplots(figsize=(12, 6))

        num_bins = 50
        log_scale = (False, True)

        x_histplot = x_stripplot
        hue_histplot = [y_stripplot[i] + '; ' + hue_stripplot[i] for i in range(len(hue_stripplot))]

        sns.histplot(x=x_histplot,
                     hue=hue_histplot,
                     bins=num_bins,
                     multiple='stack',
                     log_scale=log_scale)

        ax1.set_ylabel('Frequency')

        ax2 = ax1.twinx()
        ax2.set_ylabel('ECDF')

        sns.ecdfplot(x_histplot, ax=ax2, color='red', linestyle='dotted')

        #   Mark 5 % "center" region (w.r.t. percentiles):

        ymax_percentiles = 0.6

        percentile_47_5 = np.percentile(x_histplot, 50 - 2.5)  # 47.5% Percentile
        plt.axvline(x=percentile_47_5, color='blue', linestyle='--', ymax=ymax_percentiles)
        plt.text(x=percentile_47_5, y=ymax_percentiles, s=f'47.5th Percentile', ha='center')

        percentile_52_5 = np.percentile(x_histplot, 50 + 2.5)  # 52.5% Percentile
        plt.axvline(x=percentile_52_5, color='blue', linestyle='--', ymax=ymax_percentiles)
        plt.text(x=percentile_52_5, y=ymax_percentiles, s=f'52.5th Percentile', ha='center')

        # Vertical threshold lines
        for i in np.arange(0, 1.1, 0.1):
            plt.axvline(x=i, linestyle='dotted', color='grey')

        plt.xlabel('Predicted Probabilities')
        plt.title('Histogram - Evaluation on test data')

        plt.savefig(os.path.join(self.results_testing_path, 'histogram.png'), dpi=300)

        #   ---------------------------------------------------------------------------------------------

        #   Store predictions for consensus and no consensus cases

        consensus_cases = pd.DataFrame({
            'id': ids_consensus.tolist(),
            'prediction': preds_consensus.tolist(),
            'labels_original': trues_consensus_full.tolist()
        })

        consensus_cases.to_csv(os.path.join(self.results_testing_path, 'preds_consensus_cases.csv'), index=False)

        no_consensus_cases = pd.DataFrame({
            'id': ids_no_consensus.tolist(),
            'prediction': preds_no_consensus.tolist(),
            'labels_original': trues_no_consensus_full.tolist()
        })

        no_consensus_cases.to_csv(os.path.join(self.results_testing_path, 'preds_no_consensus_cases.csv'), index=False)

        def create_roc_curve(fpr, tpr, title: str, label: str, file_name_save: str):
            plt.figure(figsize=(12, 6))
            plt.plot(fpr, tpr, label=label)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc='lower right')

            plt.savefig(os.path.join(self.results_testing_path, file_name_save+'.png'), dpi=300)

        if self.strategy == 'baseline' or self.strategy == 'regression':

            #   ---------------------------------------------------------------------------------------------
            #   ROC curves:

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

            #   Define upper and lower threshold for decision; TODO: Tune

            neg_pred_threshold = 0.1

            pos_pred_threshold = 0.9

            #   Inbetween both thresholds -> Inconclusive

            #   ---------------------------------------------------------------------------------------------

            #   Store predictions for misclassified (fp or fn) consensus cases (given a threshold)

            #   fp consensus cases (inconclusive also counted as false)

            false_positive_indices = np.where((preds_consensus > neg_pred_threshold) & (trues_consensus_reduced == 0))[0]

            fp_samples = pd.DataFrame({
                'id': ids_consensus[false_positive_indices].tolist(),
                'prediction': preds_consensus[false_positive_indices].tolist(),
                'labels_original': trues_consensus_full[false_positive_indices].tolist(),
                'neg_pred_threshold': neg_pred_threshold
            })

            fp_samples.to_csv(os.path.join(self.results_testing_path, 'fp_samples_consensus.csv'), index=False)

            #   fn consensus cases (inconclusive also counted as false)

            false_negative_indices = np.where((preds_consensus < pos_pred_threshold) & (trues_consensus_reduced == 1))[0]

            fn_samples = pd.DataFrame({
                'id': ids_consensus[false_negative_indices].tolist(),
                'prediction': preds_consensus[false_negative_indices].tolist(),
                'labels_original': trues_consensus_full[false_negative_indices].tolist(),
                'pos_pred_threshold': pos_pred_threshold
            })

            fn_samples.to_csv(os.path.join(self.results_testing_path, 'fn_samples_consensus.csv'), index=False)

            #   ---------------------------------------------------------------------------------------------

            #   Calculate metrics precision, recall, f1-score, conf_matrix for label consensus test cases

            #   3 classes: -1 (inconclusive), 0 (normal), 1 (reduced)
            preds_consensus = np.where(preds_consensus >= pos_pred_threshold, 1,
                                       np.where(preds_consensus <= neg_pred_threshold, 0, -1))

            #   Calculate metrics per class
            average = None

            acc_score = accuracy_score(trues_consensus_reduced, preds_consensus)
            precision = precision_score(trues_consensus_reduced, preds_consensus, average=average)
            recall = recall_score(trues_consensus_reduced, preds_consensus, average=average, zero_division=0)
            f1 = f1_score(trues_consensus_reduced, preds_consensus, average=average)

            results_test_split = {
                'lower_threshold': neg_pred_threshold,
                'upper_threshold': pos_pred_threshold,
                'labels': ['inconclusive', 'normal', 'reduced'],
                'accuracy': round(acc_score, relevant_digits),
                'precision': [round(num, relevant_digits) for num in precision],
                'recall': [round(num, relevant_digits) for num in recall],
                'f1-score': [round(num, relevant_digits) for num in f1]
            }

            print(json.dumps(results_test_split, indent=4))

            with open(os.path.join(self.results_testing_path, 'perf_metrics_consensus.json'), 'w') as f:
                json.dump(results_test_split, f, indent=4)

            conf_matrix = confusion_matrix(trues_consensus_reduced, preds_consensus,
                                           labels=[-1, 0, 1])

            print('\nConfusion matrix:')
            print(conf_matrix)

            cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                                display_labels=['inconclusive', '0', '1'])
            fig, ax = plt.subplots(figsize=(12, 8))
            cm_display.plot(ax=ax)
            ax.set_title(f"Confusion Matrix for label consensus test cases \n"
                         f"(upper threshold = {pos_pred_threshold}, lower threshold = {neg_pred_threshold})")
            plt.savefig(os.path.join(self.results_testing_path, 'conf_matrix_consensus.png'), dpi=300)

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

    def __calculate_statistics(self, preds_consensus, trues_consensus_reduced, preds_no_consensus,
                               trues_no_consensus_reduced):

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

        print(statistics_on_test_split.to_string(justify='center'))

        statistics_on_test_split.to_csv(os.path.join(self.results_testing_path, 'preds_statistics.csv'))