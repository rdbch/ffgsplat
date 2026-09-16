from omegaconf import OmegaConf

from configs import Config
from trainer import Trainer


def main():
    cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    trainer = Trainer(cfg)
    trainer.build_model()
    trainer.build_optimizer()
    trainer.build_dataloaders()

    if cfg.trainer.resume:
        trainer.load_checkpoint(cfg.trainer.resume)

    trainer.train()


if __name__ == "__main__":
    main()
