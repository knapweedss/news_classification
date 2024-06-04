import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import copy
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FC_NeuralNetwork(nn.Module):
    def __init__(self, input_dim=5000):
        super(FC_NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        outp = self.model(x)
        return outp.reshape(-1)

def load_and_vectorize_data(file_path, sep='\t', max_features=5000, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path, sep=sep)

    X = df['title']
    y = df['is_fake'].values

    tfidf = TfidfVectorizer(max_features=max_features)
    X_tfidf = tfidf.fit_transform(X).toarray()
    X_tfidf = torch.tensor(X_tfidf, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=test_size, random_state=random_state)

    return X_train, X_val, y_train, y_val

def train_FC(model, loss_fn, opt, n_epochs, batch_size, lr, X_train, y_train, X_val=None, y_val=None):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = opt(model.parameters(), lr=lr)

    best_f1 = -np.inf
    best_weights = None
    loss_history = []
    roc_auc_history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        y_pred_all = []
        y_true_all = []

        for X_batch, y_batch in train_loader:
            # Forward
            y_pred = model(X_batch.to(device))
            loss = loss_fn(y_pred, y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss и F1 macro
            epoch_loss += loss.item()
            y_pred_all.extend(y_pred.round().detach().cpu().numpy())
            y_true_all.extend(y_batch.detach().cpu().numpy())

            print(f'Epoch {epoch}, Loss: {loss.item()}, F1: {f1_score(y_true_all, y_pred_all, average="macro")}')

        epoch_loss /= len(train_loader.dataset)
        loss_history.append(epoch_loss)

        # для ROC AUC
        y_pred_proba = torch.sigmoid(torch.tensor(y_pred_all)).numpy()
        roc_auc = roc_auc_score(y_true_all, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_true_all, y_pred_proba)
        roc_auc_history.append(roc_auc)

        # валидация
        if val_loader is not None:
            model.eval()
            y_pred_all_val = []
            y_true_all_val = []
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    y_val_pred = model(X_val_batch.to(device))
                    y_pred_all_val.extend(y_val_pred.round().detach().cpu().numpy())
                    y_true_all_val.extend(y_val_batch.detach().cpu().numpy())

            val_f1 = f1_score(y_true_all_val, y_pred_all_val, average='macro')
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_weights = copy.deepcopy(model.state_dict())
            if val_f1 > 0.98:  # если достигли высокого F1
                break

    # восстанавливаем модель с лучшими весами
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # ROC кривая
    plt.figure()
    plt.plot(fpr, tpr, color='coral', lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlabel("FP Rate")
    plt.ylabel("TP Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.pdf', format='pdf')

    return model, loss_history
