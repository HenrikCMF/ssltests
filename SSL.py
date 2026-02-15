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
        emb_dim: int,
        lr: float = 0.032,
        moving_average_decay: float = 0.996,
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
        self.online_encoder = online_encoder.to(self.device)
        self.target_encoder = target_encoder.to(self.device)
        self.local_epochs=local_epochs
        self.total_rounds=total_rounds
        # Target encoder: no gradients
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor head q_θ on top of online encoder output
        self.predictor = predictor.to(self.device)


        # Optimizer updates online encoder + predictor only
        #self.optimizer = torch.optim.Adam(
        #    list(self.online_encoder.parameters()) + list(self.predictor.parameters()),lr=lr,)
        total_steps = max(1, local_epochs * max(1, dataset_len))
        self.optimizer = torch.optim.SGD(
            list(self.online_encoder.parameters()) + list(self.predictor.parameters()),
            lr=0.032,
            momentum=0.9,
            weight_decay=1e-4,
        )

        # cosine over total steps
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    self.optimizer,
        #    T_max=total_steps,
        #)


        #self.m = moving_average_decay
        #self.m = 0.99
        self.m = 0.99
        self.use_ema = use_ema

    @staticmethod
    def _byol_loss(p, z):
        """
        BYOL loss: 2 - 2 * cos_sim(p, z), averaged over batch.
        p, z: [B, D]
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    @torch.no_grad()
    def _update_target(self):
        """
        EMA update of target encoder from online encoder.
        No gradients involved.
        """
        #if not self.use_ema:
        #    return
        #for p_online, p_target in zip(self.online_encoder.parameters(),
        #                              self.target_encoder.parameters()):
        #    p_target.data.mul_(self.m).add_(p_online.data, alpha=1.0 - self.m)
        for p_online, p_target in zip(self.online_encoder.parameters(),
                                  self.target_encoder.parameters()):
            #if p_target.requires_grad:  # Skip buffers
            p_target.data.mul_(self.m).add_(p_online.data, alpha=1.0 - self.m)

        for m_online, m_target in zip(self.online_encoder.modules(),
                                   self.target_encoder.modules()):
            if isinstance(m_online, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m_online.running_mean is not None:
                    m_target.running_mean.data.copy_(
                        self.m * m_target.running_mean.data + 
                        (1.0 - self.m) * m_online.running_mean.data
                    )
                if m_online.running_var is not None:
                    m_target.running_var.data.copy_(
                        self.m * m_target.running_var.data + 
                        (1.0 - self.m) * m_online.running_var.data
                    )

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
        self.target_encoder.eval()#.train()  # no gradients / BN updates
        self.dataset_len = len(train_loader)
        total_global_epochs = self.total_rounds * self.local_epochs
        current_global_epoch = (server_round - 1) * self.local_epochs  # Start of this round's "global epoch"
        progress = current_global_epoch / total_global_epochs
        total_global_steps = self.total_rounds * self.local_epochs * self.dataset_len  # dataset_len = len(train_loader)
        current_global_step = (server_round - 1) * self.local_epochs * self.dataset_len
        #print(current_global_step," / ",total_global_steps)
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

                # Online branch
                z1_online = self.online_encoder(x1, normalize=False)      # [B, D]
                z2_online = self.online_encoder(x2, normalize=False)      # [B, D]

                p1 = self.predictor(z1_online)           # [B, D]
                p2 = self.predictor(z2_online)           # [B, D]

                # Target branch (no grad)
                with torch.no_grad():
                    z1_target = self.target_encoder(x1, normalize=False)  # [B, D]
                    z2_target = self.target_encoder(x2, normalize=False)  # [B, D]

                # Symmetric BYOL loss
                loss = self._byol_loss(p1, z2_target) + self._byol_loss(p2, z1_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # EMA update of target from online (no gradient)
                current_global_step += 1
                progress = current_global_step / total_global_steps
                
                # Cosine schedule for momentum: 0.996 → 1.0
                #self.m = 1 - (1 - base_m) * (math.cos(math.pi * progress) + 1) / 2

                #progress = current_global_step / total_global_steps
                cos_lr = 0.032 * 0.5 * (1 + math.cos(math.pi * progress))
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