"""
(Command-line help shortened)

This script now performs **Masked-Language-Model pre-training**.
30 % of the input tokens are replaced by the tokenizer's ``[MASK]`` token
(override with ``--mlm_probability``).
"""

from __future__ import annotations
import argparse
from pathlib import Path

from datasets import load_dataset
from text2pic4ai import BitmapSentenceProcessor
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from pixachu._globals.logging import get_logger
from pixachu._globals.settings import PixachuSettings
from pixachu._model._trainer import (
    PixachuDataCollator,
    build_pixachu_trainer,
)


def parse_args(settings: PixachuSettings) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data / model
    p.add_argument("--dataset_name", default=settings.default_dataset_name, help="HF dataset name **or** local path")
    p.add_argument(
        "--dataset_config_name", default=settings.default_dataset_config, help="Optional additional dataset config"
    )
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_split", default="validation")
    p.add_argument("--model_name_or_path", default=settings.default_model_name)

    # training hyper-params
    p.add_argument("--output_dir", default=settings.default_output_dir)
    p.add_argument("--per_device_train_batch_size", type=int, default=settings.train_batch_size)
    p.add_argument("--per_device_eval_batch_size", type=int, default=settings.eval_batch_size)
    p.add_argument("--learning_rate", type=float, default=settings.learning_rate)
    p.add_argument("--num_train_epochs", type=float, default=settings.num_train_epochs)
    p.add_argument("--fp16", action="store_true", default=settings.fp16)
    p.add_argument(
        "--mlm_probability",
        type=float,
        default=settings.mlm_probability,
        help="Probability of masking tokens for MLM pre-training",
    )

    # misc Trainer args
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=settings.seed)

    return p.parse_args()


def main() -> None:
    settings = PixachuSettings()
    args = parse_args(settings)
    logger = get_logger()
    logger.info("Starting training with args: %s", vars(args))

    set_seed(args.seed)

    logger.info("Loading dataset %s (streaming)…", args.dataset_name)
    train_ds = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.train_split,
        streaming=True,
    ).shuffle(buffer_size=10_000, seed=args.seed)  # small, in-memory buffer

    eval_ds = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.eval_split,
        streaming=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if any(tok is None for tok in (tokenizer.mask_token, tokenizer.cls_token, tokenizer.sep_token)):
        raise ValueError("Tokenizer must define [MASK], [CLS] and [SEP] tokens")

    # Detect the text column
    possible_cols = ("text", "sentence", "content")
    text_col = next((c for c in possible_cols if c in train_ds.column_names), None)
    if text_col is None:
        raise ValueError(f"Could not locate a text column. Expected one of {possible_cols}")

    processor = BitmapSentenceProcessor()

    data_collator = PixachuDataCollator(
        processor=processor,
        tokenizer=tokenizer,
        text_key=text_col,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        evaluation_strategy="no",  # streaming ⇒ skip costly evaluation
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        report_to="none",
    )

    trainer = build_pixachu_trainer(
        model_name_or_path=args.model_name_or_path,
        train_dataset=train_ds,
        eval_dataset=None,  # evaluation disabled
        training_args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        task="mlm",
    )

    logger.info("Commencing training …")
    trainer.train()
    logger.info("Training complete. Saving final model to %s", args.output_dir)
    trainer.save_model(Path(args.output_dir))

    if eval_ds is not None:
        logger.info("Starting evaluation …")
        eval_results = trainer.evaluate(eval_dataset=eval_ds)
        logger.info("Evaluation results: %s", eval_results)


if __name__ == "__main__":
    main()
