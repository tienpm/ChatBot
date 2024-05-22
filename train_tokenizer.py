from click import group
import wandb 
import argparse
import json
import logging
import logging.config
import os
import re
from pathlib import Path

import datasets
import fire
import numpy as np
import sentencepiece as spm
import torch
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

# get date time 
import datetime
now = datetime.datetime.now()
print ("Current date and time : ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))
date_time = now.strftime("%m_%d_%H_%M")


# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="finetuning-gemma",
#     group="train_tokenizer",
#     job_type="job_train_tokenizer",
#     name=f"train_tokenizer_{date_time}",
#     # track hyperparameters and run metadata
# )

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Keep existing logging configuration
    "formatters": {
        "detailed": {
            "format": "%(asctime)s %(pathname)s:%(lineno)d %(levelname)-8s %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed",
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": True,  # Allow logging to propagate upwards if needed
        }
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger()  # get root logger


def load_and_save_dataset(dataset_hf_repo: dict, dataset_path: str):
    """
    Loads a dataset from the Hugging Face Hub and saves it to a local file.

    Args:
        dataset_hf_repo (str): The dataset repository path of the dataset on the Hugging Face Hub.
        dataset_path (str): The path where you want to save the dataset.
        split (str, optional): The specific split of the dataset to load.
                               Defaults to None, which loads all splits.
        logger_name (str, optional): The logger name which want to use
    """
    # Load the dataset from Huggingface Hub
    dataset = datasets.load_dataset(**dataset_hf_repo)

    # Save the dataset to disk. The format based on the file name extension (e.g: '.json', '.csv')
    with open(dataset_path, "a", encoding="utf-8") as f_writer:
        for item in tqdm(dataset, desc=f"Writing data to {dataset_path}"):
            f_writer.write(item["text"] + "\n")

    logger.info(f"Dataset loaded and saved to: {dataset_path}")


def has_non_alphabetic_chars(token):
    # Function to check if a token contains non-alphabetic characters
    return any(not char.isalpha() for char in token)


def train_tokenizer(
    in_text_file: str,
    sp_model_name: str = "vi-tokenizer-10K",
    max_sentence_length: int = 100000,
    vocab_size: int = 10000,
    model_type="BPE",
):
    # Build tokenizer model from input text file
    spm.SentencePieceTrainer.train(
        input=in_text_file,
        model_prefix=sp_model_name,
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=max_sentence_length,
        model_type=model_type,
        vocab_size=vocab_size,
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
        input_sentence_size = 1000000,
    )


def append_tokens(
    base_tokenizer_dir: str, new_tokenizer_model: str, new_tokenizer_dir: str
):
    # Load the base tokenizer
    base_tokenizer = LlamaTokenizer.from_pretrained(base_tokenizer_dir)
    base_sp_processor = base_tokenizer.sp_model
    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(base_sp_processor.serialized_model_proto())

    base_spm_tokens = set([p.piece for p in base_spm.pieces])

    # Load the new tokenizer model
    sp_tgt = spm.SentencePieceProcessor()
    if not new_tokenizer_model.endswith(".model"):
        new_tokenizer_model = new_tokenizer_model + ".model"
    sp_tgt.load(new_tokenizer_model)

    sp_tgt_pb2 = sp_pb2_model.ModelProto()
    sp_tgt_pb2.ParseFromString(sp_tgt.serialized_model_proto())
    new_tgt_tokens = list(set([p.piece for p in sp_tgt_pb2.pieces]))
    print("The number of original tokens:", len(base_spm_tokens))
    print("The number of new tokens:", len(new_tgt_tokens))

    # Merge the new tokens into the source tokenizer
    for piece in new_tgt_tokens:
        assert isinstance(piece, str), f"Invalid token({piece}) type {type(piece)}"
        if piece in base_spm_tokens:
            # Skip existed token.
            continue
        else:
            # Skip non-alphabetic token.
            if not has_non_alphabetic_chars(piece.replace("▁", "")):
                new_p = sp_pb2_model.ModelProto().SentencePiece()
                new_p.piece = piece
                new_p.score = 0
                base_spm.pieces.append(new_p)
            else:
                print(f"Skip non-alphabetic token {piece}")

    logger.info(f"Expand vocab from {len(base_spm_tokens)} to {len(base_spm.pieces)}")

    # Save the expanded tokenizer
    os.makedirs(new_tokenizer_dir)
    target_tokenizer_model_path = os.path.join(new_tokenizer_dir, "tokenizer.model")
    with open(file=target_tokenizer_model_path, mode="wb") as f_writer:
        f_writer.write(base_spm.SerializeToString())

    target_tokenizer = LlamaTokenizer(vocab_file=target_tokenizer_model_path)
    target_tokenizer.save_pretrained(save_directory=new_tokenizer_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train new tokenizer script")
    # Positional argument
    parser.add_argument(
        "dataset_fpath", help="The file path which save all dataset to train tokenizer"
    )
    # Optional argument
    parser.add_argument(
        "-b",
        "--base_tkn",
        default="google/gemma-7b",
        help="The directory to save the base tokenizer",
    )
    parser.add_argument(
        "-n",
        "--new_tkn",
        default="vi_gemma",
        help="The directory to save the new tokenizer trained from new dataset",
    )

    args = parser.parse_args()
    # Download and save datasets to file
    logger.info("Download and save datasets to file")
    cache_dir = Path.cwd() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_fpath = cache_dir / args.dataset_fpath

    if os.path.isfile(dataset_fpath):  # Check if it's a file (not a directory)
        os.remove(dataset_fpath)
        logger.info(f"Found the same '{args.dataset}' path. Deleted successfully.")

    list_dataset_configs = [
        {
            "path": "undertheseanlp/UTS_Text",
            "name": "base",
            "split": "train",
        }
        ,
        {
            "path": "vietgpt/binhvq_news_vi",
            "split": "train",
        },
        {
            "path": "comet24082002/vie_wiki_dataset",
            "split": "train",
        },
    ]

    for dataset_config in list_dataset_configs:
        logger.info(f"Load dataset {dataset_config['path']}")
        load_and_save_dataset(dataset_config, args.dataset_fpath)

    # Train tokenizer for Vietnameses only
    logger.info("Train tokenizer for Vietnameses only")
    pre_tokenizer_model_dir = cache_dir / "tokenizer"
    pre_tokenizer_model_dir.mkdir(parents=True, exist_ok=True)
    temp_tokenizer_model_path = pre_tokenizer_model_dir / "temp_tokenizer.model"
    if ".model" not in args.new_tkn:
        tokenizer_fname = "temp" + args.new_tkn + ".model"
        temp_tokenizer_model_path = pre_tokenizer_model_dir / tokenizer_fname
        print("temp_tokenizer_model_path =", temp_tokenizer_model_path)
    else:
        tokenizer_fname = "temp" + args.new_tkn
        temp_tokenizer_model_path = pre_tokenizer_model_dir / tokenizer_fname
        print("temp_tokenizer_model_path =", temp_tokenizer_model_path)
        
    train_tokenizer(
        in_text_file=args.dataset_fpath,
        sp_model_name=temp_tokenizer_model_path,
        max_sentence_length=100000,
        vocab_size=10000,
        model_type="BPE",
    )

    # Add token to base tokenizer of the language model and save to new directory
    logger.info(
        "Add token to base tokenizer of the language model and save to new directory"
    )
    tokenizer_model_dir = pre_tokenizer_model_dir / args.new_tkn
    append_tokens(args.base_spm, temp_tokenizer_model_path, tokenizer_model_dir)
