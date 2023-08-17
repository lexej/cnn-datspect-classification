from common import os, np, tqdm, plt, pd
from common import torch, nn


def train_model(model: nn.Module, num_epochs, train_dataloader, valid_dataloader, optimizer, loss_fn, results_path):
    best_loss = float('inf')
    best_weights = None
    best_epoch = None

    train_losses_per_epoch = []
    valid_losses_per_epoch = []

    for epoch in range(num_epochs):

        #   --------------------------------------------------------------------------------------------------

        #   1. Epoch training

        model.train()

        train_losses_per_batch = []

        #   tqdm() activates progress bar
        for batch_features, batch_labels, _ in tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}] "):

            # Forward pass
            outputs = model(batch_features)
            loss = loss_fn(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses_per_batch.append(loss.item())

        # Calculate mean epoch loss
        train_losses_per_batch_mean = float(np.mean(train_losses_per_batch))
        train_losses_per_epoch.append(train_losses_per_batch_mean)

        #   --------------------------------------------------------------------------------------------------

        #   2. Epoch validation

        model.eval()

        valid_losses_per_batch = []

        with torch.no_grad():
            for val_features, val_labels, _ in valid_dataloader:
                val_outputs = model(val_features)
                val_loss = loss_fn(val_outputs, val_labels)

                valid_losses_per_batch.append(val_loss.item())

        # Calculate mean epoch loss
        valid_losses_per_batch_mean = float(np.mean(valid_losses_per_batch))
        valid_losses_per_epoch.append(valid_losses_per_batch_mean)

        # Save weights if best
        if valid_losses_per_batch_mean < best_loss:
            best_loss = valid_losses_per_batch_mean
            best_weights = model.state_dict()
            best_epoch = epoch + 1

        # 3. Print epoch results

        print(f" Results for epoch {epoch + 1} - "
              f"train loss: {round(train_losses_per_batch_mean, 5)}, "
              f"valid loss: {round(valid_losses_per_batch_mean, 5)}")

    results_training_path = os.path.join(results_path, 'training')
    os.makedirs(results_training_path, exist_ok=True)

    best_weights_path = os.path.join(results_training_path, 'best_weights.pth')
    torch.save(best_weights, best_weights_path)

    model.load_state_dict(torch.load(best_weights_path))

    model_with_best_weights_path = os.path.join(results_training_path, 'model_with_best_weights.pth')
    torch.save(model, model_with_best_weights_path)

    #   Set evaluation mode
    model.eval()

    #   Store preds for train set and validations set on best weights

    #   Save preds for train data
    ids_train, preds_train, labels_train = __get_predictions(model=model, dataloader=train_dataloader)
    __save_preds(ids=ids_train, preds=preds_train, trues=labels_train, 
                 save_to=os.path.join(results_training_path, 'preds_train_data.csv'))
    
    #   Save preds for valid data
    ids_valid, preds_valid, labels_valid = __get_predictions(model=model, dataloader=valid_dataloader)
    __save_preds(ids=ids_valid, preds=preds_valid, trues=labels_valid, 
                 save_to=os.path.join(results_training_path, 'preds_valid_data.csv'))

    #   Create plot with loss curves

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses_per_epoch, label='train loss')
    plt.plot(range(1, num_epochs + 1), valid_losses_per_epoch, label='valid loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(results_training_path, 'training_curves.png'), dpi=300)

    return model, best_epoch, best_weights_path


def __get_predictions(model, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate preds for train set and validation set."""

    preds = []
    ids = []
    labels = []

    #   Create numpy array for predictions and ground truths

    with torch.no_grad():
        for batch in dataloader:
            batch_features, batch_labels, batch_metadata = batch

            outputs = model(batch_features)

            preds.extend(outputs.tolist())
            labels.extend(batch_labels.tolist())
            ids.extend(batch_metadata['id'].tolist())

    preds = np.array(preds).squeeze()
    ids = np.array(ids)
    labels = np.array(labels).squeeze()

    return ids, preds, labels


def __save_preds(ids: np.ndarray, preds: np.ndarray, trues: np.ndarray, save_to: str):
    """Save predictions in CSV format."""

    result = pd.DataFrame({
            'id': ids.tolist(),
            'prediction': preds.tolist(),
            'true_label': trues.tolist()
    })

    result.to_csv(save_to, index=False)

