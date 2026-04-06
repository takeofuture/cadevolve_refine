#!/usr/bin/env python3
import sys
import random
import warnings
from pathlib import Path
from functools import partial
import argparse
import yaml
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoProcessor, Qwen2VLForConditionalGeneration,
    Trainer, TrainingArguments,
)
from qwen_vl_utils import process_vision_info

from visualization import Plotter


# -----------------------------------------------------------------------------
# config utils
# -----------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout


warnings.filterwarnings("ignore", category=UserWarning, module="trimesh")


class STLImagesDataset(Dataset):
    def __init__(
        self,
        items_pkl: Path,
        max_script_len=None,
        apply_augs=False,
        language="cadevolve",
    ):
        super().__init__()
        self.plotter = None
        self.max_script_len = max_script_len
        self.apply_augs = apply_augs
        self.language = (language or "").lower()

        with open(items_pkl, "rb") as f:
            self.items = pickle.load(f)  # [(py_path_str, stl_path_str), ...]

        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __clean_code__(self, code: str):
        return "\n".join(code.split("\n")[2:-3])

    def __getitem__(self, idx):
        if self.plotter is None:
            self.plotter = Plotter()

        for _ in range(10):
            py_path_str, stl_path_str = self.items[idx]
            py_path = Path(py_path_str)
            stl_path = Path(stl_path_str)

            code = py_path.read_text(encoding="utf-8", errors="ignore")
            if self.language == "dsl":
                code = self.__clean_code__(code)

            try:
                image = self.plotter.get_img(stl_path, None, apply_augs=self.apply_augs)
                if image is None:
                    raise RuntimeError(f"Rendered image is None for {stl_path}")
                return {"image": image, "answer": code}
            except Exception as e:
                print(f"Error in visualization for {stl_path}: {e}", flush=True)
                self.plotter.reload()
                idx = random.randrange(len(self.items))

        raise RuntimeError("Too many consecutive render failures")


# -----------------------------------------------------------------------------
# label masking
# -----------------------------------------------------------------------------
def find_assistant_spans(tokenizer, ids):
    """
    Return (start_im_start, end_im_end_inclusive) spans for each assistant turn.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    asst_id = tokenizer.convert_tokens_to_ids("assistant")

    spans = []
    i = 0
    n = len(ids)
    while i < n - 2:
        if ids[i] == im_start and ids[i + 1] == asst_id:
            j = i + 2
            while j < n and ids[j] != im_end:
                j += 1
            if j < n:
                spans.append((i, j))
                i = j + 1
                continue
        i += 1
    return spans


_DEBUG_PRINTED = False


def collate_fn_for_sft(batch, processor):
    global _DEBUG_PRINTED

    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("Empty batch after filtering")

    # debug_one.py と同じ構造にそろえる
    conversations = []
    for b in batch:
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": b["image"]},
                    {"type": "text", "text": "Generate the corresponding CAD code."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": b["answer"]},
                ],
            },
        ])

    # debug_one.py と同じく、同じ conversation を
    # apply_chat_template と process_vision_info の両方に使う
    texts = [
        processor.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conv in conversations
    ]

    image_inputs, video_inputs = process_vision_info(conversations)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 型固定
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
    if "image_grid_thw" in inputs:
        inputs["image_grid_thw"] = inputs["image_grid_thw"].to(torch.long)
    if "video_grid_thw" in inputs:
        inputs["video_grid_thw"] = inputs["video_grid_thw"].to(torch.long)

    # assistant 部分だけ loss をかける
    tok = processor.tokenizer
    labels = []
    for ids in inputs["input_ids"].tolist():
        mask = [-100] * len(ids)
        for s, e in find_assistant_spans(tok, ids):
            # <|im_start|> assistant の直後から <|im_end|> まで
            mask[s + 2:e + 1] = ids[s + 2:e + 1]
        labels.append(mask)
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    # 最初の1回だけログ
    if not _DEBUG_PRINTED:
        _DEBUG_PRINTED = True
        print("=== DEBUG FIRST BATCH ===", flush=True)
        print("TEMPLATE[0]:", flush=True)
        print(texts[0], flush=True)
        print("INPUT KEYS:", list(inputs.keys()), flush=True)
        for k, v in inputs.items():
            if torch.is_tensor(v):
                print(f"{k}: dtype={v.dtype}, shape={tuple(v.shape)}", flush=True)
            else:
                print(f"{k}: type={type(v)}", flush=True)

        image_token_id = getattr(processor, "image_token_id", None)
        print("processor.image_token_id =", image_token_id, flush=True)
        print("processor.image_token =", getattr(processor, "image_token", None), flush=True)

        if image_token_id is not None:
            count = (inputs["input_ids"] == image_token_id).sum().item()
            print("count(processor.image_token_id) =", count, flush=True)

        print("DECODED INPUT[0]:", flush=True)
        print(processor.tokenizer.decode(inputs["input_ids"][0]), flush=True)
        print("========================", flush=True)

    return inputs


def run_training(cfg: dict):
    script_path = Path(__file__).resolve().parent

    # logging
    log_path = script_path / cfg["logging"]["log_path"]
    setup_logging(log_path)

    # paths
    items_pkl = Path(cfg["data"]["items_pkl"])
    model_id = cfg["model"]["model_id"]

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # seeds
    seed = int(cfg["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # processor/model
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=bool(cfg["processor"]["trust_remote_code"]),
        resized_width=int(cfg["processor"]["resized_width"]),
        resized_height=int(cfg["processor"]["resized_height"]),
        padding_side=str(cfg["processor"]["padding_side"]),
    )

    dtype_str = str(cfg["model"]["torch_dtype"]).lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {dtype_str}. Use one of: {list(dtype_map)}")
    torch_dtype = dtype_map[dtype_str]

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=str(cfg["model"]["attn_implementation"]),
        trust_remote_code=bool(cfg["model"]["trust_remote_code"]),
    )

    print("MODEL ID:", model_id, flush=True)
    print("MODEL image_token_id:", getattr(model.config, "image_token_id", None), flush=True)
    print("PROCESSOR image_token_id:", getattr(processor, "image_token_id", None), flush=True)

    # dataset
    ds_cfg = cfg["dataset"]
    train_full = STLImagesDataset(
        items_pkl=items_pkl,
        max_script_len=ds_cfg.get("max_script_len", None),
        apply_augs=bool(ds_cfg.get("apply_augs", False)),
        language=str(ds_cfg.get("language", "cadevolve")),
    )

    val_size = int(cfg["data"]["val_size"])
    idx = random.sample(range(len(train_full)), len(train_full))
    val_ds = Subset(train_full, idx[-val_size:])
    train_ds = Subset(train_full, idx[:-val_size])

    print("TRAIN_DS SIZE", len(train_ds), flush=True)
    print("VAL_DS SIZE", len(val_ds), flush=True)

    # training args
    t = cfg["training"]
    targs = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        dataloader_num_workers=int(t["dataloader_num_workers"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        lr_scheduler_type=str(t["lr_scheduler_type"]),
        num_train_epochs=float(t["num_train_epochs"]),
        warmup_steps=int(t["warmup_steps"]),
        logging_strategy=str(t["logging_strategy"]),
        logging_steps=int(t["logging_steps"]),
        save_strategy=str(t["save_strategy"]),
        save_steps=int(t["save_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        eval_strategy=str(t["eval_strategy"]),
        eval_steps=int(t["eval_steps"]),
        load_best_model_at_end=bool(t["load_best_model_at_end"]),
        bf16=bool(t["bf16"]),
        dataloader_drop_last=bool(t["dataloader_drop_last"]),
        remove_unused_columns=bool(t["remove_unused_columns"]),
        report_to=str(t["report_to"]),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn_for_sft, processor=processor),
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=bool(cfg["run"]["resume_from_checkpoint"]))
    trainer.save_model(str(output_dir / "final_model"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_training(cfg)
