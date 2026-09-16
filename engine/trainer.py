import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.logger = None

        self.train_loader = None
        self.eval_loader = None

        self.epoch = 0
        self.step = 0
        self.best_metric = None

    def build_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.cfg.model.in_dim, self.cfg.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.model.hidden_dim, self.cfg.model.out_dim),
        )
        self.model.to(self.device)

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.optimizer.lr_step)

    def build_dataloaders(self):
        train_dataset = TensorDataset(
            torch.randn(self.cfg.data.train_size, self.cfg.model.in_dim),
            torch.randn(self.cfg.data.train_size, self.cfg.model.out_dim),
        )
        eval_dataset = TensorDataset(
            torch.randn(self.cfg.data.eval_size, self.cfg.model.in_dim),
            torch.randn(self.cfg.data.eval_size, self.cfg.model.out_dim),
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.data.batch_size, shuffle=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.cfg.data.batch_size, shuffle=False)

    def train(self):
        for self.epoch in range(self.cfg.trainer.epochs):
            self.model.train()

            for batch in self.train_loader:
                metrics = self.train_step(batch)
                self.log(metrics, self.step)
                self.step += 1

            self.scheduler.step()

            if self.epoch % self.cfg.trainer.eval_every == 0:
                self.eval()

            if self.epoch % self.cfg.trainer.save_every == 0:
                self.save_checkpoint(f"checkpoints/epoch_{self.epoch}.pt")

    def train_step(self, batch):
        x, y = self._to_device(batch)

        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def eval(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                batch_metrics = self.eval_step(batch)
                total_loss += batch_metrics["loss"]
                n_batches += 1

        metrics = {"eval_loss": total_loss / n_batches}
        self.log(metrics, self.step)
        return metrics

    def eval_step(self, batch):
        x, y = self._to_device(batch)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return {"loss": loss.item()}

    def log(self, metrics, step):
        print(f"[step {step}] {metrics}")

    def save_checkpoint(self, path):
        state = {
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = state["epoch"]
        self.step = state["step"]

    def _to_device(self, batch):
        return [b.to(self.device) for b in batch]
