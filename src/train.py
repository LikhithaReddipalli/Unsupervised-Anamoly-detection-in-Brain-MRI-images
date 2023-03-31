from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.plugins import DDPPlugin,DataParallelPlugin
import hydra
import logging
from omegaconf import DictConfig
from typing import List, Optional
import wandb
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'


# Import modules / files
from src.utils import utils
#from src.datamodules.Datamodules import Brats19
from pytorch_lightning.loggers import LightningLoggerBase
import torch

log = utils.get_logger(__name__) # init logger

@hydra.main(config_path='configs', config_name='config') # Hydra decorator
def train(cfg: DictConfig) -> Optional[float]:
    if cfg.get('load_checkpoint') : # load stored checkpoint for testing or resuming training
        wandbID, checkpoints = utils.get_checkpoint(cfg, cfg.get('load_checkpoint')) # outputs a Dictionary of checkpoints and the corresponding wandb ID to resume the run

        if cfg.get('new_wandb_run',False): # If we want to onlyEvaluate a run to another wandb ID
            cfg.logger.wandb.id = wandb.util.generate_id()
            cfg.logger.wandb.note = f'corresponding original run_id: {wandbID}'
        else:
            log.info(f"Resuming wandb run")
            cfg.logger.wandb.resume = wandbID # this will allow resuming the wandb Run

    cfg.logger.wandb.group = cfg.name  # specify group name in wandb
    # Set plugins for lightning trainer
    if cfg.trainer.accelerator == 'ddp': # for better performance in ddp mode
        plugs = DDPPlugin(find_unused_parameters=True)
    else:
        plugs = None

    if "seed" in cfg: # for deterministic training (covers pytorch, numpy and python.random)
        log.info(f"Seed specified to {cfg.seed} by config")
        seed_everything(cfg.seed, workers=True)


    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule_train: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    # Init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

     # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))


    if cfg.get('load_checkpoint') : # pass checkpoint to resume from
        with open_dict(cfg):
            cfg.trainer.resume_from_checkpoint = checkpoints["fold-1"]
        log.info(f"Restoring Trainer State of loaded checkpoint")

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial", plugins=plugs
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=cfg,
        model=model,
        datamodule=datamodule_train,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if not cfg.get('onlyEval',False):

        trainer.fit(model, datamodule_train)
        validation_metrics = trainer.callback_metrics
    else:
        #cfg.get('load_checkpoint')
        #root_dir = cfg.load_checkpoint
        model.load_state_dict(torch.load(checkpoints['fold-1'])['state_dict'])


# Evaluate model on test set, using the best model achieved during training
    if cfg.get("test_after_training") and not cfg.trainer.get("fast_dev_run"):
        log.info("Starting evaluation phase!")
        preds_dict = {}
        preds_dict = {'val':{},'test':{}} # a dict for each data set
        for set in cfg.datamodule.cfg.testsets :
            # Init lightning datamodule for evaluation
            cfg.datamodule._target_ = 'src.datamodules.Datamodules.{}'.format(set)
            log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
            datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
            datamodule.setup()

            # Validation steps
            log.info("Validation of {}!".format(set))
            trainer.test(model,dataloaders=datamodule.val_dataloader())

            # evaluation results
            preds_dict['val'][set] = trainer.lightning_module.eval_dict
            log_dict = utils.summarize(preds_dict['val'][set],'val') # sets prefix val/ and removes lists for better logging in wandb

            # Test steps
            log.info("Test of {}!".format(set))
            trainer.test(model,dataloaders=datamodule.test_dataloader())

            # log to wandb
            preds_dict['test'][set] = trainer.lightning_module.eval_dict
            log_dict.update(utils.summarize(preds_dict['test'][set],'test')) # sets prefix test/ and removes lists for better logging in wandb
            log_dict = utils.summarize(log_dict,set) # sets prefix for each data set
            trainer.logger.experiment[0].log(log_dict)

        # Log RedFlag Evaluation
        log_dict_redFlag = {'val':{},'test':{}}
        log_dict_redFlag['val'] = model.redFlagEvaluation(preds_dict.copy(),cfg.healthy_key,'val') # evaluates the sample-wise detection performance for all data sets
        log_dict_redFlag['test'] = model.redFlagEvaluation(preds_dict.copy(),cfg.healthy_key,'test')
        for set in log_dict_redFlag['val']:
            trainer.logger.experiment[0].log(utils.summarize(log_dict_redFlag['val'][set],f'RedFlag/val/{set}'))
            trainer.logger.experiment[0].log(utils.summarize(log_dict_redFlag['test'][set],f'RedFlag/test/{set}'))

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    log.info(f"Best checkpoint metric:\n{trainer.checkpoint_callback.best_model_score}")

    # Return metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric",'val/Loss_comb') # default to val/loss!
    print(f'optimized metric is {optimized_metric}')
    print(validation_metrics)
    if optimized_metric and validation_metrics is not None and not cfg.get('onlyEval',False) :
        metric = validation_metrics[optimized_metric]

    if validation_metrics is not None and not cfg.get('onlyEval',False) :
        trainer.logger.experiment[0].log({optimized_metric:metric})

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=cfg,
        model=model,
        datamodule=datamodule_train,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

# TODO CLean up if onlyEval: Remove generated directory.