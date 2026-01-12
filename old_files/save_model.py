from pathlib import Path
import torch

def _ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_ssl_checkpoint(self, path, epoch):
    """
    Save encoder weights + minimal config so you can reload later.
    """
    ckpt = {
        "epoch": epoch,
        "model_class": self.model.__class__.__name__,
        "state_dict": self.model.state_dict(),          # stays on MPS/CPU fine
        "embed_dim": getattr(self.obj, "EMBED_DIM", None),
    }
    path = Path(path)
    _ensure_dir(path.parent)
    torch.save(ckpt, path)
    print(f"[ckpt] Saved SSL encoder -> {path}")

def load_ssl_encoder(path, device, model_ctor):
    """
    path: path to 'ssl-final.pt' (or epoch checkpoint)
    device: torch.device
    model_ctor: lambda emb_dim: ResNet18Proj(emb_dim)  (or ResNet9Proj)
    """
    ckpt = torch.load(path, map_location=device)
    emb_dim = ckpt.get("embed_dim", 128)
    model = model_ctor(emb_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("epoch", None)