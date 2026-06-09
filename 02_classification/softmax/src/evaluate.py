import torch


def evaluate_on_test(model, test_loader):
    """
    Evaluates the model on the test set.

    Returns:
        all_labels → true labels as numpy array
        all_preds  → predicted class indices as numpy array
        all_images → list of image tensors
    """
    model.eval()

    all_labels = []
    all_preds  = []
    all_images = []

    with torch.no_grad():
        for x, y in test_loader:
            y_hat     = model(x.view(x.size(0), -1))
            predicted = torch.argmax(y_hat, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_images.extend(x.cpu())

    return all_labels, all_preds, all_images
