from omegaconf import OmegaConf

from configs import Config
from core.trainer import Trainer


def main():
    cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    trainer = Trainer(cfg)
    trainer.build_dataloaders()
    trainer.build_model()
    trainer.build_optimizer()

    if cfg.trainer.resume:
        trainer.load_checkpoint(cfg.trainer.resume)
    trainer.build_logger()

    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
