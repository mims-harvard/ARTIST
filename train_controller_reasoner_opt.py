#!/usr/bin/env python3
"""
Train Controller-Reasoner framework with GRPO
"""
import sys
sys.path.append("/n/holylfs06/LABS/mzitnik_lab/Lab/shvat372/TS_Token_Selection/")

# ============================================
# PATCH torch.load for PyTorch 2.6 compatibility
import torch

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)  # Use False for trusted checkpoints
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
# ============================================

# import qwents_register
import os, argparse, importlib, yaml


os.environ["HF_ALLOW_CODE_EXECUTION"] = "1"


# Register custom classes
# import transformers as _tf
# # from qwents_4B_base_for_rl.modeling_qwents import QwenTSForCausalLM
# # from qwents_4B_base_for_rl.configuration_qwents import QwenTSConfig

# setattr(_tf, "QwenTSForCausalLM", QwenTSForCausalLM)
# setattr(_tf, "QwenTSConfig", QwenTSConfig)

from controller_reasoner_grpo_trainer_opt import ControllerReasonerGRPOTrainerWrapper

import warnings
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*past_key_value=None.*")


def _import_class(dotted: str):
    """Import dataset class from dotted path"""
    try:
        mod, cls = dotted.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    except Exception:
        pass
    for cand in (
        f"multimodal.{dotted}",
        f"tasks.{dotted}",
        f"multimodal_tasks.{dotted}",
    ):
        try:
            m, c = cand.rsplit(".", 1)
            return getattr(importlib.import_module(m), c)
        except Exception:
            continue
    raise ImportError(f"Could not import class from '{dotted}'")


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser("Train Controller-Reasoner with GRPO")
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    ap.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = ap.parse_args()
    
    cfg = load_yaml(args.config)
    gen = cfg.get("general", {})
    mdl = cfg.get("model", {}) or {}
    ds = cfg.get("dataset", {})
    trn = cfg.get("train", {})
    rwd = cfg.get("reward", {})
    cr_cfg = cfg.get("controller_reasoner", {})  # NEW: Controller-Reasoner specific config
    
    # Build dataset task
    class_path = ds.get("class_path") or ds.get("_target_")
    if not class_path:
        raise ValueError("dataset.class_path is required in YAML")
    init_args = ds.get("init_args", {}) or {}
    if gen.get("task_override"):
        init_args["task_name"] = gen["task_override"]
    TaskClass = _import_class(class_path)
    task = TaskClass(**init_args)
    
    # Make output dir
    out_dir = gen.get("output_dir", "./controller_reasoner_output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Build LoRA config from general section (fallback to model.lora if present)
    lora_cfg = mdl.get("lora") or {}
    # Override with general section values if present
    if gen.get("lora_r") is not None:
        lora_cfg["r"] = gen.get("lora_r")
    if gen.get("lora_alpha") is not None:
        lora_cfg["alpha"] = gen.get("lora_alpha")
    if gen.get("lora_dropout") is not None:
        lora_cfg["dropout"] = gen.get("lora_dropout")
    if gen.get("lora_target_modules") is not None:
        lora_cfg["target_modules"] = gen.get("lora_target_modules")
    
    # Build trainer
    trainer = ControllerReasonerGRPOTrainerWrapper(
        model_path=gen["model_path"],
        lightning_task=task,
        task=getattr(task, "task_name", None) or "1TS",
        training_stage=gen.get("training_stage", "mcq"),
        w_cot=gen.get("w_cot", False),
        use_lora=gen.get("use_lora", True),
        use_4bit=gen.get("use_4bit", False),
        lora_weights_path=gen.get("lora_weights_path"),
        output_dir=out_dir,
        lora_cfg=lora_cfg,
        adapter_type=mdl.get("adapter_type"),
        extra_model_cfg=mdl,
        reward_config=rwd,
        # Controller-Reasoner specific
        max_rounds=cr_cfg.get("max_rounds", 5),
        num_rollouts=cr_cfg.get("num_rollouts", 8),
        first_seg_trials=cr_cfg.get("first_seg_trials", 5),
        include_full_ts_initially=cr_cfg.get("include_full_ts_initially", True),
        use_conversation_history=cr_cfg.get("use_conversation_history", False),
        controller_temperature=cr_cfg.get("controller_temperature", 0.9),
        reasoner_temperature=cr_cfg.get("reasoner_temperature", 0.7),
        num_loops_per_generation=cr_cfg.get("num_loops_per_generation", 1),
        extended_prompt=trn.get("extended_prompt", False),
        reasoner_max_new_tokens=cr_cfg.get("reasoner_max_new_tokens", 512),
        use_uncertainty_prompt=cr_cfg.get("use_uncertainty_prompt", False),
    )
    
    # Train
    trainer.train(**trn)


if __name__ == "__main__":
    main()