import random
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv


# -------------------- Utils --------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------- Graph Construction --------------------
def create_graph(df_split, vectorizer, category_encoder, rating_scaler):
    text_features = vectorizer.transform(df_split['clean_text']).toarray()
    category_encoded = category_encoder.transform(df_split['category'])
    category_onehot = np.eye(len(category_encoder.classes_))[category_encoded]
    rating_scaled = rating_scaler.transform(df_split[['rating']])

    node_features = np.concatenate([text_features, category_onehot, rating_scaled], axis=1)

    similarity_matrix = cosine_similarity(text_features)
    edges = np.argwhere(similarity_matrix > 0.40)
    edges = edges[edges[:,0] != edges[:,1]] 
    # edge_list = []
    # n = len(df_split)
    # threshold = 0.4

    # for i in range(n):
    #     for j in range(i + 1, n):
    #         if similarity_matrix[i, j] > threshold:
    #             edge_list.append([i, j])
    #             edge_list.append([j, i])

    edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
    features = torch.tensor(node_features, dtype=torch.float)
    labels = torch.tensor(df_split['label'].values, dtype=torch.long)

    return Data(x=features, edge_index=edge_index, y=labels)


def build_graphs(df, seed):
    #set_seed(np.random.randint(0, 9999))
    set_seed(seed)

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['label'])

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectorizer.fit(train_df['clean_text'])

    category_encoder = LabelEncoder()
    category_encoder.fit(train_df['category'])

    rating_scaler = StandardScaler()
    rating_scaler.fit(train_df[['rating']])

    train_graph = create_graph(train_df, vectorizer, category_encoder, rating_scaler)
    val_graph = create_graph(val_df, vectorizer, category_encoder, rating_scaler)
    test_graph = create_graph(test_df, vectorizer, category_encoder, rating_scaler)

    return train_graph, val_graph, test_graph


# -------------------- GCN Model --------------------
class FakeReviewGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.6):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        if self.training:
            edge_index = dropout_edge(edge_index, p=0.2, training=True)[0]

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        return self.fc(x)


# -------------------- Training --------------------
def train_gcn(model, train_data, val_data, epochs=150, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_data, val_data = model.to(device), train_data.to(device), val_data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        out = model(train_data.x, train_data.edge_index)
        loss = loss_fn(out, train_data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x, val_data.edge_index)
            val_acc = (val_out.argmax(1) == val_data.y).float().mean().item()

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)

    return model


# -------------------- Evaluation --------------------
# def evaluate_gcn(model, test_graph):
#     model.eval()
#     device = next(model.parameters()).device

#     with torch.no_grad():
#         out = model(test_graph.x.to(device), test_graph.edge_index.to(device))
#         y_pred = out.argmax(1).cpu()
#         y_true = test_graph.y.cpu()
#         y_probs = F.softmax(out, dim=1).cpu().numpy()

#     return (
#         accuracy_score(y_true, y_pred),
#         f1_score(y_true, y_pred, average="weighted"),
#         roc_auc_score(y_true, y_probs[:, 1]),
#         y_true,
#         y_probs
#     )


# -------------------- Pipeline + Wrapper --------------------
# def main_pipeline(df):
#     train_graph, val_graph, test_graph = build_graphs(df, seed)
#     model = FakeReviewGCN(input_dim=train_graph.x.shape[1])
#     model = train_gcn(model, train_graph, val_graph)
#     return model, test_graph


def train_and_evaluate_gcn(df, seed):
    start_train = time()
    train_graph, val_graph, test_graph = build_graphs(df, seed)
    model = FakeReviewGCN(train_graph.x.shape[1])
    model = train_gcn(model, train_graph, val_graph)
    train_time = time() - start_train

    model.eval()
    with torch.no_grad():
        out = model(test_graph.x, test_graph.edge_index)
        y_pred = out.argmax(1).cpu()
        y_true = test_graph.y.cpu()
        y_probs = F.softmax(out, dim=1).cpu().numpy()[:,1]

    start_inf = time()
    _ = model(test_graph.x, test_graph.edge_index)
    infer_time = time() - start_inf

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    auc = roc_auc_score(y_true, y_probs)

    return acc, f1, auc, train_time, infer_time, y_true.numpy(), y_probs

