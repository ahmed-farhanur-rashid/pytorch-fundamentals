import torch
from data      import DiabetesDataset, make_splits, create_data_loader
from model     import LinearRegression
from loss_fn   import get_loss_fn
from optimizer import get_optimizer
from train     import train
from plot      import plot_losses, plot_lr_search_summary


# --- data ---
dataset = DiabetesDataset("./data/diabetes.csv", True)
train_dataset, validation_dataset, test_dataset = make_splits(dataset)
train_loader = create_data_loader(train_dataset, 32, True)
validation_loader = create_data_loader(validation_dataset, len(validation_dataset))
test_loader = create_data_loader(test_dataset, len(test_dataset))


# --- tuning ---
LR_CANDIDATES = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
train_losses_across_lrs = {}
validation_losses_across_lrs = {}
models = {}


for lr in LR_CANDIDATES:
    model = LinearRegression(dataset.n_features, 1)
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(model, lr)

    model, train_losses, validation_losses = train(
        model, train_loader, loss_fn, optimizer,
        epoch=100, validation_loader=validation_loader
    )

    train_losses_across_lrs[lr] = train_losses
    validation_losses_across_lrs[lr] = validation_losses
    models[lr] = model


# --- finding best model ---
best_lr = min(validation_losses_across_lrs,
            key = lambda lr : validation_losses_across_lrs[lr][-1] # finding best lr based on validation loss
)

best_model = models[best_lr]
best_train_loss = train_losses_across_lrs[best_lr]
best_validation_loss = validation_losses_across_lrs[best_lr]


# --- plotting ---
plot_losses(
    losses    = {
        "train"      : best_train_loss,
        "validation" : best_validation_loss
    },
    title     = "Best Model Loss",
    save_path = "../visualization/loss.png"
)

plot_lr_search_summary(
    learning_rates = LR_CANDIDATES,
    train_errors   = [train_losses_across_lrs[lr][-1] for lr in LR_CANDIDATES],
    val_errors     = [validation_losses_across_lrs[lr][-1]   for lr in LR_CANDIDATES],
    save_path      = "../visualization/lr_search.png"
)