
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

import seaborn as sns
import numpy as np
from scipy.special import expit 


def count_labels(data_loader):
    tako, normal = 0, 0
    for batch in data_loader:
        labels = batch["label"]
        tako += (labels == 1).sum().item()
        normal += (labels == 0).sum().item()
    return tako, normal

def plot_roc_curve(y_true, y_prob, auc_score, timestamp):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{timestamp} | ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/{timestamp}_roc_curve.png", dpi=300, bbox_inches='tight')  # Save before plt.show()

    plt.show()

def plot_confusion_matrix(y_true, y_pred, timestamp, labels=["Normal", "Takotsubo"]):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(5, 4))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{timestamp} | Confusion Matrix")
    plt.grid(False)
    plt.savefig(f"plots/{timestamp}_conf_matr.png", dpi=300, bbox_inches='tight')  # Save before plt.show()

    plt.show()

def plot_loss_curves(train_losses, val_losses, timestamp):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{timestamp} | Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{timestamp}_loss_curves.png", dpi=300, bbox_inches='tight')  # Save before plt.show()

    plt.show()

def plot_calibration_curve(y_true, y_prob, timestamp, title="Calibration Curve", label="Model"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker='o', label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title(f"{timestamp} | {title}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{timestamp}_{title}.png", dpi=300, bbox_inches='tight')  # Save before plt.show()

    plt.show()

def plot_view_attention_weights(model,timestamp, view_names=["Axial", "Sagittal", "Coronal"]):
    """
    Plots the softmax-normalized attention weights assigned to each view.
    
    Parameters:
    - model (MultiViewCNN): The trained model with `view_weights` as a parameter.
    - view_names (list of str): Names of the views corresponding to the weights.
    """
    # Ensure view_weights exists
    if not hasattr(model, "view_weights"):
        raise AttributeError("The model does not have 'view_weights'. Make sure attention is implemented.")

    # Apply softmax to get importance scores
    with torch.no_grad():
        weights = F.softmax(model.view_weights, dim=0).cpu().numpy()

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.bar(view_names, weights, color="skyblue")
    plt.title("Attention Weights Across Views")
    plt.ylabel("Importance (Softmax Score)")
    plt.ylim(0, 1)
    for i, w in enumerate(weights):
        plt.text(i, w + 0.02, f"{w:.2f}", ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"plots/{timestamp}_att_weights.png", dpi=300, bbox_inches='tight')  # Save before plt.show()

    plt.show()
    
def plot_precision_recall_curve(y_true, y_scores, timestamp, title="Precision-Recall Curve", label=None, color="blue"):
    """
    Plots a precision-recall curve for binary classification.

    Parameters:
    - y_true: Ground-truth binary labels (array-like, shape = [n_samples])
    - y_scores: Predicted scores or probabilities (array-like, shape = [n_samples])
    - title: Title of the plot
    - label: Optional label for the legend (e.g., "Model A")
    - color: Line color
    - save_path: If provided, saves the figure to this file path
    """

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'{label or "Model"} (AP = {ap:.2f})', color=color)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{timestamp} | {title}")
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{timestamp}_prec_rec.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_logits_and_probs(logits, labels, timestamp, title_prefix=""):
    logits = np.array(logits).flatten()
    labels = np.array(labels).flatten()
    probs = expit(logits)  # Apply sigmoid

    plt.figure(figsize=(12, 5))

    # Plot logits
    plt.subplot(1, 2, 1)
    sns.histplot(logits[labels == 0], label="Normal", color="blue", kde=True, stat="density")
    sns.histplot(logits[labels == 1], label="Takotsubo", color="red", kde=True, stat="density")
    plt.axvline(0, color='gray', linestyle='--')
    plt.title(f"{timestamp} | {title_prefix}Logits Distribution")
    plt.xlabel("Logit Value")
    plt.legend()

    # Plot probabilities
    plt.subplot(1, 2, 2)
    sns.histplot(probs[labels == 0], label="Normal", color="blue", kde=True, stat="density")
    sns.histplot(probs[labels == 1], label="Takotsubo", color="red", kde=True, stat="density")
    plt.axvline(0.5, color='gray', linestyle='--')
    plt.title(f"{timestamp} | {title_prefix}Sigmoid Probabilities")
    plt.xlabel("Predicted Probability")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/{timestamp}_logits.png", dpi=300, bbox_inches='tight')  # Save before plt.show()

    plt.show()