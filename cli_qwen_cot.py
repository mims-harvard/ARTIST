import logging
import os
import warnings

warnings.filterwarnings("ignore")
import sys
from pprint import pprint
from typing import Any, Optional

import pandas as pd
import torch
from dotenv import dotenv_values
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, Callback)
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.states import TrainerFn
from torch import autograd

import sys, os
sys.path.insert(0, os.getcwd())

from utils_general import get_logger

torch.set_float32_matmul_precision('medium')

logger = get_logger(__name__)
CONFIG = dotenv_values(".env")

os.environ.update(CONFIG)

class DebugOptimizerParams(Callback):
    def on_train_start(self, trainer, pl_module):
        opt = trainer.optimizers[0]
        names = []
        for group in opt.param_groups:
            for p in group['params']:
                # find the matching name in the module
                for n, param in pl_module.named_parameters():
                    if param is p:
                        names.append(n)
        print("\n>>> OPTIMIZER WILL UPDATE:")
        for n in sorted(set(names)):
            print(f"    {n}")

class AutoLoRASaver(Callback):
    def __init__(self, save_every_n_steps=1000):
        self.save_every_n_steps = save_every_n_steps
        self.last_saved_step = 0


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Save LoRA weights periodically and clean memory"""
        current_step = trainer.global_step
        
        # Memory cleanup every 50 steps
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        # # Periodic LoRA saving
        # if (current_step - self.last_saved_step >= self.save_every_n_steps and 
        #     hasattr(pl_module, 'use_lora') and pl_module.use_lora):
            
        #     try:
        #         # Create step-based directory
        #         step_dir = os.path.join(trainer.default_root_dir, f"lora_step_{current_step}")
                
        #         if hasattr(pl_module, 'save_all_weights'):
        #             pl_module.save_all_weights(step_dir)
        #         else:
        #             pl_module.save_lora_weights(step_dir)
                
        #         self.last_save_step = current_step
        #         logger.info(f"LoRA weights saved at step {current_step}: {step_dir}")
                
        #     except Exception as e:
        #         logger.error(f"Failed to save LoRA weights at step {current_step}: {e}")


    """Automatically saves LoRA weights for every checkpoint"""
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Only save if using LoRA
        # if not (hasattr(pl_module, 'use_lora') and pl_module.use_lora):
        #     return
        
        checkpoint_callback = trainer.checkpoint_callback
        if not checkpoint_callback:
            return
        
        # Save for every checkpoint by using current epoch
        current_epoch = trainer.current_epoch
        lora_dir = os.path.join(checkpoint_callback.dirpath, f"lora_weights_epoch_{current_epoch:02d}")

        if not os.path.exists(lora_dir):
            try:
                # Use save_all_weights for exp2 models, save_lora_weights for others
                if hasattr(pl_module, 'save_all_weights') and pl_module.use_lora == False:
                    pl_module.save_all_weights(lora_dir)
                    logger.info(f"All weights saved for epoch {current_epoch}: {lora_dir}")
                elif hasattr(pl_module, 'save_lora_weights') and pl_module.use_lora == True:
                    pl_module.save_lora_weights(lora_dir)
                    logger.info(f"LoRA weights saved for epoch {current_epoch}: {lora_dir}")
            except Exception as e:
                logger.error(f"Failed to save weights for epoch {current_epoch}: {e}")
   
    def on_train_end(self, trainer, pl_module):
        """Training ended - LoRA weights already saved every epoch"""
        if not (hasattr(pl_module, 'use_lora') and pl_module.use_lora):
            return
        
        logger.info("Training completed. LoRA weights have been saved for each epoch.")


def add_general_args(parent_parser):
    """ Adds arguments that aren't part of pl.Trainer, but are useful """
    parent_parser.add_argument("--checkpoint_metric", type=str, default=None,
                                help="Metric to optimize for during training")
    parent_parser.add_argument("--checkpoint_mode", type=str, default="max",
                                help="Metric direction to optimize for during training")
    parent_parser.add_argument("--no_wandb", default=False, action="store_true",
                                help="Run without wandb logging")
    parent_parser.add_argument("--notes", type=str, default=None,
                                help="Notes to be sent to WandB")
    parent_parser.add_argument("--early_stopping_patience", type=int, default=None,
                                help="path to validation dataset")
    parent_parser.add_argument("--gradient_log_interval", default=0, type=int,
                                help = "Interval with which to log gradients to WandB. 0 -> Never")
    parent_parser.add_argument("--load_weights_path", default=None, type=str)
    parent_parser.add_argument("--freeze_encoder", action="store_true", default=False)
    parent_parser.add_argument("--run_name", type=str, default=None,
                                help="run name to use for to WandB")
    parent_parser.add_argument("--pl_seed", type=int, default=2494,
                                help="Pytorch Lightning seed for current experiment")
    parent_parser.add_argument("--no_ckpt", action="store_true", default=False,
                                help="Don't save any model checkpoints")
    parent_parser.add_argument("--default_root_dir", type=str, default="lightning_logs",
                                help="Root directory for saving logs and checkpoints")
    parent_parser.add_argument("--test_checkpoint_path", type=str, default=None,
                                help="Path to checkpoint for testing (required for standalone test)")
    parent_parser.add_argument("--training_stage", type=str, default="mcq", 
                                choices=["mcq", "alignment", "reasoning"],
                                help="Training stage: 'alignment' for open-ended QA, 'mcq' for multiple choice")
    parent_parser.add_argument("--model_version", type=str, default="qwen3_ts_model", 
                                choices=["qwen2_vl_zoom","qwen3_ts_model", "qwen3_model_w_cot_sft","qwen3_ts_exp2", "qwenvl_3B"],
                                help="Model version to use: 'qwen3_ts_model' for base model, 'qwen3_ts_exp2' for experimental model")
    parent_parser.add_argument("--random_segment_selection", action="store_true", default=False,
                                help="Use only random segment selection for the model")
    parent_parser.add_argument("--total_ts", action="store_true", default=False,
                                help="Use total time series for the model")
    parent_parser.add_argument("--regular_sft", action="store_true", default=False,
                                help="Use regular SFT for the model")
    parent_parser.add_argument("--cot_no_tools", action="store_true", default=False,
                                help="Use cot without tools for the model")
    # New: adapter config path for model.adapter_config
    parent_parser.add_argument("--adapter_config_path", type=str, default=None,
                                help="Path to YAML/JSON file containing adapter_config dict for the model")
    parent_parser.add_argument("--save_top_k", type=int, default=-1,
                                help="Number of best checkpoints to keep (-1: keep all, 1: keep only the best)")
    parent_parser.add_argument("--no_last_ckpt", action="store_true", default=False,
                                help="If set, do not save the 'last.ckpt' checkpoint")
    # Special tokens control
    parent_parser.add_argument("--use_ts_start_end", action="store_true", default=False,
                                help="Use <ts_start> and <ts_end> markers with TS embeddings inserted between. If not set, use single <ts> token.")
    # Force loading full checkpoint (.ckpt) instead of LoRA weights for testing
    parent_parser.add_argument("--use_full_checkpoint", action="store_true", default=False,
                            help="Force loading full checkpoint (.ckpt) instead of LoRA weights for testing")
    # ---- pass@k eval knobs ----
    parent_parser.add_argument("--num_samples_per_question", type=int, default=1,
                                help="Number of independent generations per question (n) for pass@k.")
    parent_parser.add_argument("--passk", type=str, default="1,2,4,8",
                                help="Comma-separated k values to report. Only k <= n are used.")
    parent_parser.add_argument("--max_eval_samples", type=int, default=-1,
                                help="Limit the number of dataset items evaluated (runtime cap). None = no cap.")
    parent_parser.add_argument(
    "--eval_passk",
    action="store_true",
    default=False,
    help="If set, run pass@k evaluation (multiple samples per question). Otherwise run standard single-sample eval."
    )
    parent_parser.add_argument(
        "--balanced_sampler",
        action="store_true",
        default=False,
        help="Use weighted sampling to balance class frequencies in the TRAIN loader."
    )
    return parent_parser

class WandBSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:

        if isinstance(trainer.logger, WandbLogger):
            # If we're at rank zero and using WandBLogger then we probably want to
            # log the config
            log_dir = trainer.logger.experiment.dir
            fs = get_filesystem(log_dir)

            config_path = os.path.join(log_dir, self.config_filename)
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=True, multifile=self.multifile
            )
        else:
            super().setup(trainer,pl_module,stage=stage)

class CLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.batch_size","data.batch_size",apply_on="parse")
        parser.link_arguments("data.task_name", "model.task", apply_on="parse")
        parser.add_optimizer_args(torch.optim.Adam)
        add_general_args(parser)
        # Link training_stage after it's been added by add_general_args
        parser.link_arguments("training_stage", "model.training_stage", apply_on="parse")
        # Link random_segment_selection to model
        parser.link_arguments("random_segment_selection", "model.random_segment_selection", apply_on="parse")
        # Link total_ts to model
        parser.link_arguments("total_ts", "model.total_ts", apply_on="parse")
        parser.link_arguments("cot_no_tools", "model.cot_no_tools", apply_on="parse")
        # Link regular_sft to model
        parser.link_arguments("regular_sft", "model.regular_sft", apply_on="parse")
        # pass@k / runtime cap eval flags -> model
        parser.link_arguments("num_samples_per_question", "model.num_samples_per_question", apply_on="parse")
        parser.link_arguments("passk", "model.passk", apply_on="parse")
        parser.link_arguments("max_eval_samples", "model.max_eval_samples", apply_on="parse")
        parser.link_arguments("eval_passk", "model.eval_passk", apply_on="parse")
        parser.link_arguments("balanced_sampler", "data.balanced_sampler", apply_on="parse")
        # Link w_cot parameter between data and model
        # parser.link_arguments("data.w_cot", "model.w_cot", apply_on="parse")
        # Link special tokens flag to model
        # parser.link_arguments("use_ts_start_end", "model.use_ts_start_end", apply_on="parse")
        
    @staticmethod
    def get_datamodule_class(training_stage: str, task_name: str):
        """Get the appropriate datamodule class based on training stage"""
        from multimodal import MultimodalMCQTask, MultimodalOpenTask, TimerbedECGTask, TimerbedRCWTask, TSQAOpenTask
        
        if training_stage == "alignment":
            print("Using MultimodalOpenTask for alignment training (open-ended QA)")
            return MultimodalOpenTask
        elif task_name == 'ECG':
            print("Using TimerbedECGTask for ECG training")
            return TimerbedECGTask
        elif task_name == 'RCW':
            print("Using TimerbedRCWTask for RCW training")
            return TimerbedRCWTask
        # elif task_name == 'TSQA':
        #     print("Using TSQAOpenTask for TSQA training")
        #     return TSQAOpenTask 
        else:
            print("Using MultimodalMCQTask for MCQ training")
            return MultimodalMCQTask

    def get_checkpoint_path_for_test(self, config_test_checkpoint_path: str, default_root_dir: str) -> Optional[str]:
        """
        Get the checkpoint path for testing, with fallback options
        """
        # Option 1: Explicit checkpoint path provided
        if config_test_checkpoint_path:
            if os.path.exists(config_test_checkpoint_path):
                logger.info(f"Using specified checkpoint: {config_test_checkpoint_path}")
                return config_test_checkpoint_path
            else:
                raise FileNotFoundError(f"Specified checkpoint not found: {config_test_checkpoint_path}")
        
        # Option 2: Try to read from best_checkpoint.txt
        best_checkpoint_file = os.path.join(default_root_dir, "best_checkpoint.txt")
        if os.path.exists(best_checkpoint_file):
            with open(best_checkpoint_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Best checkpoint:"):
                        checkpoint_path = line.split("Best checkpoint:")[1].strip()
                        if os.path.exists(checkpoint_path):
                            logger.info(f"Using checkpoint from best_checkpoint.txt: {checkpoint_path}")
                            return checkpoint_path
        
        # Option 3: Look for checkpoints in the standard location
        checkpoint_dir = os.path.join(default_root_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            import glob
            best_checkpoints = glob.glob(os.path.join(checkpoint_dir, "best-*.ckpt"))
            if best_checkpoints:
                # Use the most recent best checkpoint
                best_checkpoint = max(best_checkpoints, key=os.path.getmtime)
                logger.info(f"Found best checkpoint: {best_checkpoint}")
                return best_checkpoint
            
            # Fall back to any .ckpt file
            all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
            if all_checkpoints:
                latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
                logger.info(f"Using latest checkpoint: {latest_checkpoint}")
                return latest_checkpoint
        
        logger.warning("No checkpoint found for testing!")
        return None
    
    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        subcommand = self.config.subcommand
        extra_callbacks = []

        pl_seed = self.config[subcommand]["pl_seed"]
        seed_everything(pl_seed)

        
        checkpoint_metric = self.config[subcommand]["checkpoint_metric"]
        mode = self.config[subcommand]["checkpoint_mode"]
        run_name = self.config[subcommand]["run_name"]
        default_root_dir = self.config[subcommand]["default_root_dir"]
        
        # Add timestamp to checkpoint directory to prevent overwrites
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(default_root_dir, f"checkpoints_{timestamp}")
        
        if self.config.subcommand == TrainerFn.FITTING:
            if self.datamodule.val_dataloader() is not None:
            
                if self.datamodule.is_classification:
                    if checkpoint_metric is None:            
                        checkpoint_metric = "val/roc_auc"
                        mode = "max"
                else:
                    if checkpoint_metric is None:            
                        checkpoint_metric = "val/loss"
                        mode = "min"
                
                if self.config["fit"]["early_stopping_patience"]:
                    early_stopping_callback = EarlyStopping(monitor=checkpoint_metric,
                                                            patience=self.config["fit"]["early_stopping_patience"],
                                                            mode=mode)
                    extra_callbacks.append(early_stopping_callback)
            else:
                if checkpoint_metric is None:            
                    checkpoint_metric = "train/loss"
                    mode = "min"
            
            if (not self.config[subcommand]["no_ckpt"]) and (not os.environ.get("NO_CKPT",None)):
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_last = not self.config[subcommand]["no_last_ckpt"]
                save_top_k = self.config[subcommand]["save_top_k"]

                self.checkpoint_callback = ModelCheckpoint(
                                    dirpath=checkpoint_dir,
                                    filename='best-{epoch:02d}-{' + checkpoint_metric.replace('/', '_') + ':.4f}',
                                    save_last=save_last,
                                    save_top_k=save_top_k,
                                    save_on_train_epoch_end=True,
                                    monitor=checkpoint_metric,
                                    # every_n_epochs=1,
                                    mode=mode,
                                    auto_insert_metric_name=False
                                    )
            
                extra_callbacks.append(self.checkpoint_callback)
    
        # Always add LearningRateMonitor regardless of WandB
        lr_monitor = LearningRateMonitor(logging_interval='step')
        extra_callbacks.append(lr_monitor)
        
        if not self.config[self.config.subcommand]["no_wandb"]:
            import wandb            
            if os.environ.get("WANDB_DIR",None):
                save_dir = os.environ.get("WANDB_DIR")
            else:
                save_dir = "."

            data_logger = WandbLogger(project=CONFIG["WANDB_PROJECT"],
                                name=run_name,
                                config=dict(self.config.as_dict()[self.config.subcommand]),
                                notes=self.config[self.config.subcommand]["notes"],
                                log_model=False, #saves checkpoints to wandb as artifacts, might add overhead 
                                save_dir=save_dir,
                                )   #id of run to resume from, None if model is not from checkpoint. Alternative: directly use id = model.logger.experiment.id, or try setting WANDB_RUN_ID env variable                
           
            if not callable(data_logger.experiment.summary):
                data_logger.experiment.summary["task"] = self.datamodule.get_name()
                data_logger.experiment.summary["model"] = self.model.name
                data_logger.experiment.config.update(self.model.hparams, allow_val_change=True)
                # self.model.save_hyperparameters() 
                
                # Necessary to save config in the right location
                data_logger._save_dir = data_logger.experiment.dir
            
            # self.model.wandb_id = data_logger.version 

        else:
            data_logger = None
        kwargs["logger"] = data_logger

        extra_callbacks = extra_callbacks + [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        kwargs["default_root_dir"] = self.config[subcommand]["default_root_dir"]
        trainer_config = {**self._get(self.config_init, "trainer"), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)
    
    def instantiate_classes(self) -> None:
        if self.config.subcommand == "test":
            # Get the test checkpoint path from config (our custom argument)
            custom_test_checkpoint_path = getattr(self.config.test, 'test_checkpoint_path', None)
            default_root_dir = getattr(self.config.test, 'default_root_dir', 'lightning_logs')
            use_full_checkpoint = getattr(self.config.test, 'use_full_checkpoint', False)
            
            # check if their is a lora model
            model_config = self.config.test.model
            is_lora_model = hasattr(model_config, 'lora_weights_path') and model_config.lora_weights_path

            if is_lora_model and not use_full_checkpoint:
                logger.info("LoRA model detected, will not load weights from instantiate_classes")
                pass
        
            else:
                # If no explicit checkpoint path, try to find one
                if not custom_test_checkpoint_path:
                    found_checkpoint = self.get_checkpoint_path_for_test(None, default_root_dir)
                    if found_checkpoint:
                        logger.info(f"Auto-discovered checkpoint: {found_checkpoint}")
                        self.config.test.ckpt_path = found_checkpoint
                        custom_test_checkpoint_path = found_checkpoint

                else:
                    if not os.path.exists(custom_test_checkpoint_path):
                        raise FileNotFoundError(f"Specified checkpoint not found: {custom_test_checkpoint_path}")
                    logger.info(f"Using specified checkpoint: {custom_test_checkpoint_path}")
                    self.config.test.ckpt_path = custom_test_checkpoint_path
                    self.config.test.model.lora_weights_path = None # clear lora weights path to force full checkpoint loading
                    self.config.test.model.external_itformer_patchtst_dir_path = None
                    # self.config.test.model.use_lora = False
        super().instantiate_classes()
        
        # Set test_checkpoint_path on the model after instantiation for epoch extraction
        if self.config.subcommand == "test":
            custom_test_checkpoint_path = getattr(self.config.test, 'test_checkpoint_path', None)
            if not custom_test_checkpoint_path:
                custom_test_checkpoint_path = getattr(self.config.test, 'ckpt_path', None)
            
            if custom_test_checkpoint_path and hasattr(self.model, 'test_checkpoint_path'):
                self.model.test_checkpoint_path = custom_test_checkpoint_path

    def before_fit(self):
        # Enables logging of gradients to WandB
        gradient_log_interval = self.config["fit"]["gradient_log_interval"]
        if isinstance(self.trainer.logger, WandbLogger) and gradient_log_interval:
            self.trainer.logger.watch(self.model, log="all", log_freq=gradient_log_interval)
        
        if self.config["fit"]["load_weights_path"]:
            state_dict = torch.load(self.config["fit"]["load_weights_path"])["state_dict"]
            self.model.load_state_dict(state_dict,strict=False)

        if self.config["fit"]["freeze_encoder"]:
            self.model.freeze_encoder()

    def after_fit(self):
        if self.trainer.is_global_zero:
            logger.info(f"Best model score: {self.checkpoint_callback.best_model_score}")
            logger.info(f"Best model path: {self.checkpoint_callback.best_model_path}")
        results = {}

        if self.trainer.state.fn == TrainerFn.FITTING:
            if (
                    self.trainer.checkpoint_callback
                    and self.trainer.checkpoint_callback.best_model_path
            ):
                ckpt_path = self.trainer.checkpoint_callback.best_model_path
                # Disable useless logging
                logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(
                    logging.WARNING
                )
                logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(
                    logging.WARNING
                )

                self.trainer.callbacks = []

                
                test_dataloader = self.trainer.datamodule.test_dataloader()
                if test_dataloader:
                    fn_kwargs = {
                        "model": self.model,
                        "dataloaders": [test_dataloader],
                        "ckpt_path": ckpt_path,
                        "verbose": False,
                    }
                    results = self.trainer.test(**fn_kwargs)[0]
                else:
                    results = {}

                if hasattr(self.model, "wandb_id") and results:
                    self.model.upload_predictions_to_wandb()

        else:
            results = self.trainer.logged_metrics

        if results:
            pprint(results)

    def set_defaults(self):
        ...

if __name__ == "__main__":
    # Check for training_stage in command line arguments
    training_stage = "alignment" 
    for i, arg in enumerate(sys.argv):
        if arg == "--training_stage" and i + 1 < len(sys.argv):
            training_stage = sys.argv[i + 1]
            break
        elif arg.startswith("--training_stage="):
            training_stage = arg.split("=", 1)[1]
            break
    
    # Check for model_version in command line arguments
    model_version = "qwen3_ts_model"  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--model_version" and i + 1 < len(sys.argv):
            model_version = sys.argv[i + 1]
            break
        elif arg.startswith("--model_version="):
            model_version = arg.split("=", 1)[1]
            break
    
    # Check for task_name in command line arguments
    task_name = "ETI"  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--data.task_name" and i + 1 < len(sys.argv):
            task_name = sys.argv[i + 1]
            break
        elif arg.startswith("--data.task_name="):
            task_name = arg.split("=", 1)[1]
            break
    

    from ts_model_w_cot_sft import Qwen3TSLightning as QwenTSLightning
    print("Using Qwen3TS Base Model")
    
    # Get appropriate datamodule class
    datamodule_class = CLI.get_datamodule_class(training_stage, task_name)

    trainer_defaults = dict(
                        accelerator="cuda",
                        num_sanity_val_steps=0,
                        devices=-1,
                        profiler=None,
                        callbacks=[AutoLoRASaver(save_every_n_steps=70000)], #, DebugOptimizerParams()
              )
   
    cli = CLI(model_class=QwenTSLightning,                 # ← your LightningModule subclass
        datamodule_class=datamodule_class,  # ← dynamically chosen based on training_stage
        trainer_defaults=trainer_defaults,
            save_config_kwargs={"overwrite": True},
            save_config_callback=SaveConfigCallback, #WandBSaveConfigCallback
          )

