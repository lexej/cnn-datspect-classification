from common import np, DataLoader


def dataloader_to_ndarrays(dataloader: DataLoader, squeeze: bool = False):

    X, y, ids = [], [], []

    for batch in dataloader:
        batch_features, batch_labels, batch_metadata = batch

        X.extend(batch_features.cpu().numpy())
        y.extend(batch_labels.cpu().numpy())
        ids.extend(batch_metadata['id'].cpu().numpy())
    
    X = np.array(X)
    y = np.array(y)
    ids = np.array(ids)

    if squeeze:
        X = np.squeeze(X)
        y = np.squeeze(y)
        ids = np.squeeze(ids)

    return X, y, ids

