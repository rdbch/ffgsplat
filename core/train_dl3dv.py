from omegaconf import OmegaConf

from core.trainer_dl3dv import DL3DVTrainConfig, TrainerDL3DV


def main():
    cfg = OmegaConf.structured(DL3DVTrainConfig)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    trainer = TrainerDL3DV(cfg)
    trainer.build_dataloaders()
    trainer.build_model()
    trainer.build_optimizer()

    if cfg.trainer.resume:
        trainer.load_checkpoint(cfg.trainer.resume)
    elif cfg.trainer.init_from:
        trainer.load_weights(cfg.trainer.init_from)
    trainer.build_logger()

    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
