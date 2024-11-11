from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %%
def prepare_dataset(dataframe, policy, class_names):
    dataset_df = dataframe[
        dataframe["Frontal/Lateral"] == "Frontal"
    ]  # take frontal pics only
    # Check which images had a support device. This is optional.
    # dataset_df = dataset_df[dataset_df["Support Devices"] == 1].reset_index(drop=True)
    df = dataset_df.sample(
        frac=1.0, random_state=1
    )  # If desired, downsample the dataset.
    df.fillna(0, inplace=True)  # fill the with zeros
    x_path, y_df = df["Path"].to_numpy(), df[class_names]
    class_ones = ["Atelectasis", "Cardiomegaly"]
    y = np.empty(y_df.shape, dtype=int)
    for i, (index, row) in enumerate(y_df.iterrows()):
        labels = []
        for cls in class_names:
            curr_val = row[cls]
            feat_val = 0
            if curr_val:
                curr_val = float(curr_val)
                if curr_val == 1:
                    feat_val = 1
                elif curr_val == -1:
                    if policy == "ones":
                        feat_val = 1
                    elif policy == "zeroes":
                        feat_val = 0
                    elif policy == "mixed":
                        if cls in class_ones:
                            feat_val = 1
                        else:
                            feat_val = 0
                    else:
                        feat_val = 0
                else:
                    feat_val = 0
            else:
                feat_val = 0

            labels.append(feat_val)

        y[i] = labels

    x_path = ["./" + x for x in x_path]

    return x_path, y

# %%
def roc_curves(class_names, all_outputs, all_targets):
    auc_per_class = {}
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        # Calculate FPR, TPR, and thresholds for each class
        fpr, tpr, _ = roc_curve(all_targets[:, i], all_outputs[:, i])
        roc_auc = auc(fpr, tpr)  # Calculate AUC for each class
        auc_per_class[class_name] = roc_auc

        # Plot ROC curve for the current class
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    # Plot configuration
    plt.plot([0, 1], [0, 1], "k--")  # Dashed diagonal line for random guessing
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Each Class")
    plt.legend(loc="lower right")
    plt.show()

    average_auc = np.nansum(list(auc_per_class.values())) / len(auc_per_class)
    print(f"\nAverage AUC-ROC across classes: {average_auc:.4f}")
    return auc_per_class


# %%
def confusion_matrices(class_names, all_outputs, all_targets):
    # Convert probabilities to binary predictions (threshold = 0.5)
    binary_predictions = (np.array(all_outputs) > 0.5).astype(int)

    for i, class_name in enumerate(class_names):
        # Calculate confusion matrix for each class
        cm = confusion_matrix(all_targets[:, i], binary_predictions[:, i])

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        plt.title(f"Confusion Matrix for {class_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()


# %%
def plot_param_kde(model):
    """Plots KDEs for parameter distributions of each layer in the model on a single plot."""
    plt.figure(figsize=(10, 6))
    for name, param in model.named_parameters():
        if param.requires_grad:
            sns.kdeplot(
                param.data.cpu().numpy().flatten(),
                label=name,
                fill=True,
                common_norm=False,
            )
    plt.title("Parameter Distribution KDEs (All Layers)")
    plt.xlabel("Parameter Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# %%
def plot_avg_update_to_weight_ratio_kde(model, optimizer):
    """Calculates and plots the average update-to-weight ratio across layers in the model."""
    layer_ratios = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Calculate weight update (gradient * learning rate)
            for group in optimizer.param_groups:
                lr = group["lr"]
            update = param.grad * lr

            # Calculate the update-to-weight ratio for the layer
            update_to_weight_ratio = (
                (update / (param + 1e-8)).cpu().detach().numpy().flatten()
            )

            # Calculate the mean update-to-weight ratio for the layer
            avg_ratio = update_to_weight_ratio.mean()
            layer_ratios.append(avg_ratio)

    overall_mean = np.log10(np.abs(np.nansum(layer_ratios) / len(layer_ratios)))

    # Plot KDE for the average update-to-weight ratios across layers
    plt.figure(figsize=(10, 6))
    sns.kdeplot(np.log10(np.abs(layer_ratios)), fill=True)
    plt.axvline(
        overall_mean, color="red", linestyle="--", label=f"Mean: {overall_mean:.4f}"
    )
    plt.title("Average Update-to-Weight Ratio Distribution Across Layers")
    plt.xlabel("Average Update-to-Weight Ratio")
    plt.ylabel("Density")
    plt.show()
    print(overall_mean)
