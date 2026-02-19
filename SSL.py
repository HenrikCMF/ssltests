import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
class SimpleBYOL:
    """
    Minimal BYOL trainer:

    - online_encoder: model to update with gradients
    - target_encoder: model used as stop-grad target (no gradients)
    - predictor: small MLP on top of online encoder output
    - update_target() does EMA update of target from online (no gradients)

    Expects a dataloader that yields (x1, x2) batches of augmented views.
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        lr: float = 0.032,
        moving_average_decay: float = 0.99,
        device: str = None,
        use_ema: bool = True,
        local_epochs: int = 1,
        dataset_len: int = 0,
        total_rounds: int =100
    ):
        if torch.cuda.is_available():
            device="cuda"
        elif torch.backends.mps.is_available():
            device="mps"
        else:
            device="cpu"
        self.device=device
        # Store encoders
        self.dataset_len=dataset_len
        self.online_encoder = online_encoder.to(self.device)
        self.target_encoder = target_encoder.to(self.device)
        self.predictor = predictor.to(self.device)
        self.local_epochs=local_epochs
        self.total_rounds=total_rounds
        self.lr=lr
        # Target encoder: no gradients
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor head q_θ on top of online encoder output
        self.optimizer = torch.optim.SGD(
            list(self.online_encoder.parameters()) + list(self.predictor.parameters()),#+list(self.target_encoder.parameters()),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0005,
        )
        self.m = moving_average_decay
        self.use_ema = use_ema

    @staticmethod
    def _byol_loss(p, z):
        """
        BYOL loss: 2 - 2 * cos_sim(p, z), averaged over batch.
        p, z: [B, D]
        """
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        return 2 - 2 * (p * z).sum(dim=-1)

    @torch.no_grad()
    def _update_target(self):
        """
        EMA update of target encoder from online encoder.
        No gradients involved.
        """
        m = self.m
        alpha = 1.0 - m

        # 1. EMA on all parameters (weights + biases)
        for p_online, p_target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            #p_target.data.mul_(m).add_(p_online.data, alpha=alpha)
            p_target.data=p_target.data * self.m + (1 - self.m) * p_online.data

        # 2. EMA on all floating-point buffers (i.e. running_mean & running_var)
        #    We skip num_batches_tracked because it's a counter (long dtype)
        #for b_online, b_target in zip(
        #    self.online_encoder.buffers(),
        #    self.target_encoder.buffers()
        #):
        #    if b_online.dtype.is_floating_point:          # only mean/var
        #        b_target.data.mul_(m).add_(b_online.data, alpha=alpha)

    def train(self, train_loader, epochs: int, server_round):
        """
        Run BYOL training for `epochs` over the given loader.

        train_loader must yield either:
            v1, v2               (no labels)
        or:
            (v1, v2), labels     (labels will be ignored)
        """
        self.online_encoder.train()
        self.predictor.train()
        #self.target_encoder.eval()#.train()  # no gradients / BN updates
        #cos_lr = self.lr * 0.5 * (1 + math.cos(math.pi * progress))
        #self.set_lr(cos_lr)
        #print(current_global_step," / ",total_global_steps)
        total_global_steps = self.total_rounds * self.local_epochs * self.dataset_len
        current_global_step = (server_round - 1) * self.local_epochs * self.dataset_len
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                # Unpack batch (handle optional labels)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x1, x2 = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                    # e.g. ( (x1, x2), y )
                    (x1, x2), _ = batch
                else:
                    raise ValueError("Expected batch to be (x1, x2) or ((x1, x2), y).")

                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                # Online branch
                z1_online = self.online_encoder(x1, normalize=False)      # [B, D]
                z2_online = self.online_encoder(x2, normalize=False)      # [B, D]

                p1 = self.predictor(z1_online)           # [B, D]
                p2 = self.predictor(z2_online)           # [B, D]

                # Target branch (no grad)
                with torch.no_grad():
                    z1_target = self.target_encoder(x1, normalize=False)  # [B, D]
                    z2_target = self.target_encoder(x2, normalize=False)  # [B, D]
                    z1_target = z1_target.detach()
                    z2_target = z2_target.detach()

                # Symmetric BYOL loss
                loss = (self._byol_loss(p1, z2_target) + self._byol_loss(p2, z1_target)).mean()

                
                loss.backward()
                self.optimizer.step()

                # EMA update of target from online (no gradient)
                current_global_step += 1
                #progress = current_global_step / total_global_steps
                

                progress = current_global_step / total_global_steps
                cos_lr = self.lr * 0.5 * (1 + math.cos(math.pi * progress))
                self.set_lr(cos_lr)
                self._update_target()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)

        # After training, the updated model is self.online_encoder
        return avg_loss
    
    def set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    # Example usage
    online_net = models.resnet18(num_classes=128)
    target_net = models.resnet18(num_classes=128)
    byol_trainer = SimpleBYOL(online_net, target_net, emb_dim=128, device=None)

    # Dummy dataloader
    from torch.utils.data import DataLoader, TensorDataset
    x_dummy = torch.randn(100, 3, 224, 224)
    dataset = TensorDataset(x_dummy, x_dummy)  # Just dummy data
    dataloader = DataLoader(dataset, batch_size=16)

    loss = byol_trainer.train(dataloader, epochs=1)
    print(f"Training loss: {loss}")