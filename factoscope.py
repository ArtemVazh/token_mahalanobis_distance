import os
import copy
import numpy as np
import torch
import random

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.generation_metrics.aggregated_metric import AggregatedMetric

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from torchvision import models
from transformers import set_seed

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GRUNet(nn.Module):
    def __init__(self, emb_dim, input_dim=1, hidden_dim1=128, hidden_dim2=64, feature_dim=32, dropout=0.5):
        super(GRUNet, self).__init__()

        # First GRU layer
        self.gru1 = nn.GRU(input_size=input_dim,
                           hidden_size=hidden_dim1, batch_first=True)

        # Second GRU layer
        self.gru2 = nn.GRU(input_size=hidden_dim1,
                           hidden_size=hidden_dim2, batch_first=True)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Dense layers for feature extraction
        self.fc1 = nn.Linear(hidden_dim2, feature_dim)
        self.fc2 = nn.Linear(feature_dim, emb_dim)  # This is the feature layer

    def forward(self, x):
        x = x.squeeze(1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        # We take the output from the last time step of the GRU
        x = self.dropout(x[:, -1, :])
        x = self.fc1(x)
        x = self.fc2(x)  # This is the output from the feature layer

        # x = F.softmax(x, dim=1)

        return x


class TriDataset(torch_data.Dataset):
    """Custom Dataset for loading and processing the data."""

    def __init__(self, features, labels, source=None, transform=None, test_stage=False):
        self.features = features
        self.labels = labels
        self.source = source
        self.transform = transform
        self.test_stage = test_stage

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        anchor = self.features[index]
        anchor_label = self.labels[index]

        # positive
        pos_index = random.randint(0, len(self.labels) - 1)
        while self.labels[pos_index] != anchor_label:
            pos_index = random.randint(0, len(self.labels) - 1)
        positive = self.features[pos_index]
        positive_label = self.labels[pos_index]

        # negative
        neg_index = random.randint(0, len(self.labels) - 1)
        while (self.labels[neg_index] == anchor_label) and (not self.test_stage):
            neg_index = random.randint(0, len(self.labels) - 1)
        negative = self.features[neg_index]
        negative_label = self.labels[neg_index]

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return torch.from_numpy(anchor).float(), \
            torch.from_numpy(positive).float(), \
            torch.from_numpy(negative).float(), \
            torch.tensor(anchor_label, dtype=torch.long), \
            torch.tensor(positive_label, dtype=torch.long), \
            torch.tensor(negative_label, dtype=torch.long)

    def get_source(self):
        return self.source


class CombinedTriNet(nn.Module):
    def __init__(self, act, grunet, embdistance, prob, emb_dim, feature_dim=64):
        super(CombinedTriNet, self).__init__()
        self.act = act
        self.grunet = grunet
        self.embdistance = embdistance
        self.prob = prob

        self.fc1 = nn.Linear(emb_dim*4, feature_dim)

    def forward(self, x, act_dim):

        x_activation, x_rank, x_embdis, x_prob = x[:, :, :, :act_dim], x[:, :, :, act_dim:(
            act_dim+1)], x[:, :, :, (act_dim+1):(act_dim+11)], x[:, :, :, (act_dim+11):]

        x1 = self.act(x_activation)
        x2 = self.grunet(x_rank)
        x3 = self.embdistance(x_embdis)
        x4 = self.prob(x_prob)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.fc1(x))
        embedding = F.normalize(x, p=2, dim=1)

        return embedding
    
    
class ActRankTriNet(nn.Module):
    def __init__(self, act, grunet, emb_dim, feature_dim=64):
        super(ActRankTriNet, self).__init__()
        self.act = act
        self.grunet = grunet

        self.fc1 = nn.Linear(emb_dim*2, feature_dim)

    def forward(self, x, act_dim):

        x_activation, x_rank = x[:, :, :, :act_dim], x[:, :, :, act_dim:(act_dim+1)]
        
        x1 = self.act(x_activation)
        x2 = self.grunet(x_rank)

        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.fc1(x))
        embedding = F.normalize(x, p=2, dim=1)

        return embedding

class ActRankTopkIdTriNet(nn.Module):
    def __init__(self, act, grunet, embdistance, emb_dim, feature_dim=64):
        super(ActRankTopkIdTriNet, self).__init__()
        self.act = act
        self.grunet = grunet
        self.embdistance = embdistance

        self.fc1 = nn.Linear(emb_dim*3, feature_dim)

    def forward(self, x, act_dim):

        x_activation, x_rank, x_embdis = x[:, :, :, :act_dim], x[:, :, :, act_dim:(act_dim+1)], x[:, :, :, (act_dim+1):(act_dim+11)]
        
        x1 = self.act(x_activation)
        x2 = self.grunet(x_rank)
        x3 = self.embdistance(x_embdis)

        x = torch.cat((x1, x2, x3), dim=1)

        x = F.relu(self.fc1(x))
        embedding = F.normalize(x, p=2, dim=1)

        return embedding
        
class ActTriNet(nn.Module):
    def __init__(self, act):
        super(ActTriNet, self).__init__()
        self.act = act

    def forward(self, x, act_dim):

        x_activation= x[:, :, :, :act_dim]
        x = self.act(x_activation)
        embedding = F.normalize(x, p=2, dim=1)

        return embedding
    
class RankTriNet(nn.Module):
    def __init__(self, grunet):
        super(RankTriNet, self).__init__()
        self.grunet = grunet

    def forward(self, x, act_dim):

        x_rank= x[:, :, :, act_dim:(act_dim+1)]
        x = self.grunet(x_rank)
        embedding = F.normalize(x, p=2, dim=1)

        return embedding

class TopkIdTriNet(nn.Module):
    def __init__(self, embdistance):
        super(TopkIdTriNet, self).__init__()
        self.embdistance = embdistance

    def forward(self, x, act_dim):

        x_embdis= x[:, :, :, (act_dim+1):(act_dim+11)]
        x = self.embdistance(x_embdis)
        embedding = F.normalize(x, p=2, dim=1)

        return embedding
    
class TopkProbTriNet(nn.Module):
    def __init__(self, prob):
        super(TopkProbTriNet, self).__init__()
        self.prob = prob

    def forward(self, x, act_dim):

        x_prob= x[:, :, :, (act_dim+11):]
        x = self.prob(x_prob)
        embedding = F.normalize(x, p=2, dim=1)

        return embedding
    
def test_model(model, test_loader, support_loader, act_dim, squeeze_dim=1):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []

    support_set_labels = []
    support_set_output = []
    with torch.no_grad():
        for i, (support_data, _, _, support_label, _, _) in enumerate(support_loader):
            support_data = support_data.unsqueeze(squeeze_dim).cuda()
            if i == 0:
                support_set_output = model(support_data, act_dim)
                support_set_labels = support_label
            else:
                support_set_output = torch.cat(
                    (support_set_output, model(support_data, act_dim)), dim=0)
                support_set_labels = torch.cat(
                    (support_set_labels, support_label), dim=0)

    # compare the distance between the embedding of the test image and the embedding of the support set
    with torch.no_grad():
        for i, (anchor, _, _, anchor_label, _, _) in enumerate(test_loader):
            anchor = anchor.unsqueeze(squeeze_dim).cuda()
            anchor_embedding = model(anchor, act_dim)
            anchor_embedding = anchor_embedding.squeeze()
            dist = F.pairwise_distance(
                anchor_embedding, support_set_output, p=2)
            pred = support_set_labels[torch.argmin(dist, -1)]
            y_pred.append(int(pred))
            y_true.append(int(anchor_label))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = conf_matrix.ravel()
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    print(f"True Positives: {TP / (TP + FN)}, False Positives: {FP / (FP + TN)}, True Negatives: {TN / (TN + FP)}, False Negatives: {FN / (TP + FN)}")
    print(f"Accuracy: {accuracy}")
    return accuracy, y_pred

def predict_model(model, test_loader, support_loader, act_dim, squeeze_dim=1, support_set_output=None, support_set_labels=None, return_dist=False):
    model.eval()
    y_pred = []

    if (support_set_output is None) or (support_set_labels is None):
        support_set_labels = []
        support_set_output = []
        with torch.no_grad():
            for i, (support_data, _, _, support_label, _, _) in enumerate(support_loader):
                support_data = support_data.unsqueeze(squeeze_dim).cuda()
                if i == 0:
                    support_set_output = model(support_data, act_dim)
                    support_set_labels = support_label
                else:
                    support_set_output = torch.cat(
                        (support_set_output, model(support_data, act_dim)), dim=0)
                    support_set_labels = torch.cat(
                        (support_set_labels, support_label), dim=0)

    # compare the distance between the embedding of the test image and the embedding of the support set
    with torch.no_grad():
        for i, (anchor, _, _, anchor_label, _, _) in enumerate(test_loader):
            anchor = anchor.unsqueeze(squeeze_dim).cuda()
            anchor_embedding = model(anchor, act_dim)
            anchor_embedding = anchor_embedding.squeeze()
            dist = F.pairwise_distance(anchor_embedding, support_set_output, p=2)
            if return_dist:
                correct_dist = dist[support_set_labels == 1]
                if len(correct_dist):
                    y_pred.append(correct_dist.min().item())
                else:
                    y_pred.append(1 - dist.min().item())
            else:
                pred = support_set_labels[torch.argmin(dist, -1)]
                y_pred.append(int(pred))
    return y_pred, support_set_labels, support_set_output
    
def train_model(model, train_loader, dev_loader, support_loader, act_dim, squeeze_dim=1, epochs=30):
    seed_everything(42)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    highest_acc = 0
    best_model = model
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (anchor, positive, negative, _, _, _) in enumerate(train_loader):

            anchor = anchor.unsqueeze(squeeze_dim).cuda()
            positive = positive.unsqueeze(squeeze_dim).cuda()
            negative = negative.unsqueeze(squeeze_dim).cuda()
            optimizer.zero_grad()
            anchor_embedding = model(anchor, act_dim)
            positive_embedding = model(positive, act_dim)
            negative_embedding = model(negative, act_dim)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, train_loss))
        test_accuracy, _ = test_model(model, dev_loader, support_loader, act_dim, squeeze_dim=1)
        if test_accuracy > highest_acc:
            highest_acc = test_accuracy
            best_model = copy.deepcopy(model)
    return best_model

def process_activation_data(all_data, mean=None, std=None):
    if mean is None:
        mean = np.mean(all_data)
    if std is None:
        std = np.std(all_data)
    all_data = (all_data - mean) / std
    return all_data, mean, std

def process_rank_data(all_rank):
    a = -1
    all_rank = 1 / (a * (all_rank - 1) + 1 + 1e-7)
    return all_rank

class LLMFactoscope(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        metric_thr: float = 0.0,
        hidden_layers: List[int] = [0, -1],
        metric = None,
        metric_name: str = "",
        aggregated: bool = False,
        emb_dim: int = 24,
        topk: int = 10,
        return_dist: bool = False,
    ):
        self.hidden_layers = hidden_layers
        dependencies = ["train_greedy_tokens", "train_target_texts", "final_output_ranks", "topk_layer_distance", "topk_prob",
                        "train_final_output_ranks", "train_topk_layer_distance", "train_topk_prob"]
        for layer in self.hidden_layers:
            if layer == -1:
                dependencies += ["token_embeddings", "train_token_embeddings"]
            else:
                dependencies += [f"token_embeddings_{layer}", f"train_token_embeddings_{layer}"]
        super().__init__(dependencies, "sequence")
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregated = aggregated
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)
        self.metric_name = metric_name
        self.embeddings_type=embeddings_type
        self.emb_dim = emb_dim
        self.topk = topk
        self.return_dist = return_dist

    def __str__(self):
        dist = "_dist" if self.return_dist else ""
        return f"Factoscope{dist}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted: 
            seed_everything(42)
            train_greedy_texts = stats[f"train_greedy_texts"]
            train_greedy_tokens = stats[f"train_greedy_tokens"]
            train_target_texts = stats[f"train_target_texts"]
   
            metrics = []
            for x, y, x_t in zip(train_greedy_texts, train_target_texts, train_greedy_tokens):
                if isinstance(y, list) and (not self.aggregated):
                    y_ = y[0]
                elif isinstance(y, str) and (self.aggregated):
                    y_ = [y]
                else:
                    y_ = y
                if self.metric_name == "Accuracy":
                    metrics.append(self.metric({"greedy_texts": [x], "target_texts": [y_]}, [y_], [y_])[0])
                else:
                    metrics.append(int(self.metric({"greedy_texts": [x], "target_texts": [y_]}, [y_], [y_])[0] > self.metric_thr))
            self.seq_metrics = np.array(metrics)
            
            #n_instances, n_layers
            final_output_ranks = np.array(stats[f"train_final_output_ranks"])
            #n_instances, n_layers, topk
            topk_tokens_distance = np.array(stats[f"train_topk_layer_distance"])
            #n_instances, n_layers, topk
            topk_prob = np.array(stats[f"train_topk_prob"])
            embeddings = []
            for layer in self.hidden_layers:
                layer_embeddings = []
                if layer == -1:
                    train_token_embeddings = stats[f"train_token_embeddings_{self.embeddings_type}"]
                else:
                    train_token_embeddings = stats[f"train_token_embeddings_{self.embeddings_type}_{layer}"]
                k = 0
                for tokens in train_greedy_tokens:
                    layer_embeddings.append(train_token_embeddings[k])
                    k += len(tokens)
                embeddings.append(np.array(layer_embeddings))
            #n_instances, n_layers, embed_dim
            embeddings = np.asarray(embeddings).transpose(1, 0, 2)
            
            train_indices, dev_indices = train_test_split(list(range(len(self.seq_metrics))), test_size=0.2, random_state=42)
            train_indices, support_indices = train_test_split(train_indices, test_size=0.2, random_state=42)
           
            train_seq_metrics = self.seq_metrics[train_indices]
            support_seq_metrics = self.seq_metrics[support_indices]
            dev_seq_metrics = self.seq_metrics[dev_indices]
            
            final_output_ranks = final_output_ranks.reshape(len(self.seq_metrics), -1)[:, self.hidden_layers]
            topk_tokens_distance = topk_tokens_distance.reshape(len(self.seq_metrics), -1, self.topk)[:, self.hidden_layers]
            topk_prob = topk_prob.reshape(len(self.seq_metrics), -1, self.topk)[:, self.hidden_layers]
                         
            train_final_output_ranks = final_output_ranks[train_indices][:, :, None]
            train_topk_tokens_distance = topk_tokens_distance[train_indices]
            train_topk_prob = topk_prob[train_indices]
            train_embeddings = embeddings[train_indices]
            
            support_final_output_ranks = final_output_ranks[support_indices][:, :, None]
            support_topk_tokens_distance = topk_tokens_distance[support_indices]
            support_topk_prob = topk_prob[support_indices]
            support_embeddings = embeddings[support_indices]
            
            dev_final_output_ranks = final_output_ranks[dev_indices][:, :, None]
            dev_topk_tokens_distance = topk_tokens_distance[dev_indices]
            dev_topk_prob = topk_prob[dev_indices]
            dev_embeddings = embeddings[dev_indices]

            train_embeddings_processed, self.mean, self.std = process_activation_data(train_embeddings)
            train_final_output_ranks_processed = process_rank_data(train_final_output_ranks)
            
            support_embeddings_processed, _, _ = process_activation_data(support_embeddings, self.mean, self.std)
            support_final_output_ranks_processed = process_rank_data(support_final_output_ranks)
            
            dev_embeddings_processed, _, _ = process_activation_data(dev_embeddings, self.mean, self.std)
            dev_final_output_ranks_processed = process_rank_data(dev_final_output_ranks)

            prob_resnet_model = models.resnet18(pretrained=False, num_classes=self.emb_dim).train().cuda()
            prob_resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()

            emb_dist_resnet_model = models.resnet18(pretrained=False, num_classes=self.emb_dim).train().cuda()
            emb_dist_resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()
            
            grunet_model = GRUNet(emb_dim=self.emb_dim).train().cuda()

            act_resnet_model = models.resnet18(pretrained=False, num_classes=self.emb_dim).cuda()
            act_resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()
            
            combined_model = CombinedTriNet(act_resnet_model, grunet_model, emb_dist_resnet_model, prob_resnet_model, emb_dim=self.emb_dim).cuda()  
          
            train_data = np.concatenate((train_embeddings_processed, train_final_output_ranks_processed,
                                         train_topk_tokens_distance, train_topk_prob), axis=2)
            
            support_data = np.concatenate((support_embeddings_processed, support_final_output_ranks_processed,
                                           support_topk_tokens_distance, support_topk_prob), axis=2)
            
            dev_data = np.concatenate((dev_embeddings_processed, dev_final_output_ranks_processed,
                                       dev_topk_tokens_distance, dev_topk_prob), axis=2)
            
            train_dataset = TriDataset(train_data, train_seq_metrics)
            support_dataset = TriDataset(support_data, support_seq_metrics)
            dev_dataset = TriDataset(dev_data, dev_seq_metrics)
                        
            train_loader = torch_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            self.support_loader = torch_data.DataLoader(support_dataset, batch_size=64, shuffle=False)
            dev_loader = torch_data.DataLoader(dev_dataset, batch_size=1, shuffle=False)
            self.ue_predictor = train_model(combined_model, train_loader, dev_loader, self.support_loader, train_embeddings_processed.shape[-1])
            self.support_set_labels, self.support_set_output = None, None
            self.is_fitted = True
        
        greedy_tokens = stats[f"greedy_tokens"]
        batch_size = len(greedy_tokens)
        #n_instances, n_layers
        final_output_ranks = np.array(stats[f"final_output_ranks"])
        final_output_ranks = final_output_ranks.reshape(batch_size, -1)[:, self.hidden_layers]
        final_output_ranks = final_output_ranks[:, :, None]
        #n_instances, n_layers, topk
        topk_tokens_distance = np.array(stats[f"topk_layer_distance"])
        topk_tokens_distance = topk_tokens_distance.reshape(batch_size, -1, self.topk)[:, self.hidden_layers]
        #n_instances, n_layers, topk
        topk_prob = np.array(stats[f"topk_prob"])
        topk_prob = topk_prob.reshape(batch_size, -1, self.topk)[:, self.hidden_layers]

        embeddings = []
        for layer in self.hidden_layers:
            layer_embeddings = []
            if layer == -1:
                token_embeddings = stats[f"token_embeddings_{self.embeddings_type}"]
            else:
                token_embeddings = stats[f"token_embeddings_{self.embeddings_type}_{layer}"]
            k = 0
            for tokens in greedy_tokens:
                layer_embeddings.append(token_embeddings[k])
                k += len(tokens)
            embeddings.append(np.array(layer_embeddings))
        #n_instances, n_layers, embed_dim
        embeddings = np.asarray(embeddings).transpose(1, 0, 2)
        
        embeddings_processed, _, _ = process_activation_data(embeddings, self.mean, self.std)
        final_output_ranks_processed = process_rank_data(final_output_ranks)
        
        data = np.concatenate((embeddings_processed, final_output_ranks_processed,
                               topk_tokens_distance, topk_prob), axis=2)
        dataset = TriDataset(data, np.zeros(len(data)), test_stage=True)
        loader = torch_data.DataLoader(dataset, batch_size=1, shuffle=False)

        y_pred, self.support_set_labels, self.support_set_output = predict_model(self.ue_predictor, loader, self.support_loader, 
                                                                                 embeddings_processed.shape[-1],
                                                                                 support_set_labels=self.support_set_labels, support_set_output=self.support_set_output,
                                                                                 squeeze_dim=1, return_dist=self.return_dist)
        if self.return_dist:
            return np.array(y_pred)
        return 1 - np.array(y_pred)