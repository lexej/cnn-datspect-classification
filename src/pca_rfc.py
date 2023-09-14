from common import os, plt, PCA, RandomForestClassifier, RANDOM_SEED
from common import np, accuracy_score, recall_score, precision_score
from common import pickle
from common import DataLoader

from helper import dataloader_to_ndarrays

from evaluation import _save_preds

from data import _choose_label_from_available_labels


def __create_pca_components_plot(pca_components, num_cols, save_to_path):

    num_rows = int(np.ceil(len(pca_components) / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))

    for i, ax in enumerate(axes.ravel()):
        if i < len(pca_components):
            ax.imshow(pca_components[i]) # , cmap='gray'
            ax.set_title(f"PCA component {i + 1}")
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_to_path, dpi=300)
    plt.close(fig)  


def fit_rfc(train_dataloader: DataLoader, num_pca_components, results_path):

    results_training_path = os.path.join(results_path, 'training')
    os.makedirs(results_training_path, exist_ok=True)

    X_train, y_train, ids_train = dataloader_to_ndarrays(train_dataloader, squeeze=True)

    img_dim_original = (X_train.shape[-2], X_train.shape[-1])

    #   Flatten ndarray of images to shape (num_samples, num_features)

    X_train = X_train.reshape(len(X_train), -1)
    
    #   Fit PCA model on train split and transform the subset

    pca = PCA(n_components=num_pca_components, random_state=RANDOM_SEED)
    pca.fit(X_train)

    #   Transform splits 

    X_train_pca_transformed = pca.transform(X_train)

    #   Access the detected PCA components

    pca_components = pca.components_.reshape((num_pca_components, 
                                              img_dim_original[0], 
                                              img_dim_original[1]))

    #   Create plot of pca components
    __create_pca_components_plot(pca_components, num_cols=5, 
                                 save_to_path=os.path.join(results_training_path, 'pca_components.png'))

    rfc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)

    rfc.fit(X=X_train_pca_transformed, y=y_train)

    preds_train = rfc.predict_proba(X_train_pca_transformed)

    #   Reduce to predictions of prob of class 1
    preds_train = preds_train[:, 1]

    # preds_train_thresholded = rfc.predict(X_train_pca_transformed)

    #   Save rfc and pca model

    rfc_model_path = os.path.join(results_training_path, 'rfc_trained.pkl')
    with open(rfc_model_path, 'wb') as f:
        pickle.dump(rfc, f)

    pca_model_path = os.path.join(results_training_path, 'pca_trained.pkl')
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca, f)
    
    #   Save preds for train split

    _save_preds(ids=ids_train, preds=preds_train, trues=y_train, 
                save_to=os.path.join(results_training_path, 'preds_train_data.csv'))

    return rfc_model_path, pca_model_path


def evaluate_rfc(rfc_path, pca_path, test_dataloader: DataLoader, strategy, save_to_path: str):

    with open(rfc_path, 'rb') as f_rfc:
        rfc = pickle.load(f_rfc)

    with open(pca_path, 'rb') as f_pca:
        pca = pickle.load(f_pca)

    X_test, y_test, ids_test = dataloader_to_ndarrays(test_dataloader, squeeze=True)

    X_test = X_test.reshape(len(X_test), -1)

    X_test_pca_transformed = pca.transform(X_test)

    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_reduced = []

        for i in y_test:
            chosen_label = _choose_label_from_available_labels(i,
                                                               label_selection_strategy='majority',
                                                               strategy=strategy)
            y_test_reduced.append(chosen_label.cpu().numpy())
        
        y_test_reduced = np.array(y_test_reduced)
    else:
        y_test_reduced = y_test

    preds_test = rfc.predict_proba(X_test_pca_transformed)
    preds_test = preds_test[:, 1]

    preds_test_thresholded = rfc.predict(X_test_pca_transformed)

    acc_test = accuracy_score(y_true=y_test_reduced, y_pred=preds_test_thresholded)
    recall_test = recall_score(y_true=y_test_reduced, y_pred=preds_test_thresholded)
    precision_test = precision_score(y_true=y_test_reduced, y_pred=preds_test_thresholded)

    # images_reconstructed = pca.inverse_transform(images_transformed_pca).reshape(original_shape)

    #   Save preds for test data

    _save_preds(ids=ids_test, preds=preds_test, trues=y_test, save_to=save_to_path)


"""

########################################################################
# Visualize 

test_img_idx = 10

plt.figure(figsize=(10, 5))

# Plot the first image on the left subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.imshow(images_transformed[test_img_idx])  # Use 'cmap' to specify the color map (e.g., 'gray')
plt.title('Original')


# Plot the second image on the right subplot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.imshow(images_reconstructed[test_img_idx])  # Use 'cmap' to specify the color map (e.g., 'gray')
plt.title('Reconstructed after PCA')

plt.tight_layout()  # Adjust spacing between subplots for a clean layout
plt.show()


########################################################################

"""