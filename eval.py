import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from architectures import ResNet18Projv2
import flwr as fl
from flwr.common import parameters_to_ndarrays
from dataloader import build_eval_loaders
@torch.no_grad()
def extract_feats(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        z = encoder.encode_backbone(x, normalize=True)  # [B,512], normalized
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats, 0), torch.cat(labels, 0)

@torch.no_grad()
def knn_acc(train_feats, train_labels, test_feats, test_labels, k=200, temperature=0.1):
    # cosine sim because features are L2-normalized
    sims = test_feats @ train_feats.T                    # [Ntest, Ntrain]
    topk_sims, topk_idx = sims.topk(k=k, dim=1)
    topk_labels = train_labels[topk_idx]

    weights = torch.exp(topk_sims / temperature)
    num_classes = int(train_labels.max().item()) + 1

    votes = torch.zeros(test_feats.size(0), num_classes)
    for c in range(num_classes):
        votes[:, c] = (weights * (topk_labels == c).float()).sum(dim=1)

    pred = votes.argmax(dim=1)
    return (pred == test_labels).float().mean().item()



class FedAvgWithKnnEval(fl.server.strategy.FedAvg):
    def __init__(self, *args, data_dir="./data", device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)
        self.eval_model = ResNet18Projv2(emb_dim=128).to(self.device)

        self.train_ld, self.test_ld = build_eval_loaders(
            data_dir=data_dir, batch_size=512, num_workers=2
        )

        # Cache labels/features per eval call? simplest is recompute each round.

    def _set_model_from_global_params(self, ndarrays):
        # You are federating (model params + predictor params).
        # For eval you only need the model params.
        n_model = sum(1 for _ in self.eval_model.parameters())
        model_params = ndarrays[:n_model]
        with torch.no_grad():
            for p, w in zip(self.eval_model.parameters(), model_params):
                p.copy_(torch.from_numpy(w).to(p.device))

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is None:
            return None

        parameters_aggregated, metrics_aggregated = aggregated

        # Run kNN eval on aggregated global model
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        self._set_model_from_global_params(ndarrays)

        train_feats, train_labels = extract_feats(self.eval_model, self.train_ld, self.device)
        test_feats, test_labels   = extract_feats(self.eval_model, self.test_ld,  self.device)
        acc = knn_acc(train_feats, train_labels, test_feats, test_labels, k=200, temperature=0.1)

        print(f"[Server] round={server_round} kNN acc={acc*100:.2f}%")

        # optionally report to Flower metrics
        metrics_aggregated = metrics_aggregated or {}
        metrics_aggregated["knn_acc"] = acc

        return parameters_aggregated, metrics_aggregated
