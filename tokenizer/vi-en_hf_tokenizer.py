import logging
import logging.config

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import AutoTokenizer

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


def getTrainingCorpus(data, step=1000):
    for start_idx in range(0, len(data), step):
        samples = data[start_idx : start_idx + step]
        yield samples["text"]


def finetuneTokenizer(
    tokenizer_path, sample_loader, data, batch, vocab_size, save_path
):
    """
    Retrain tokenizer from a pretrained tokenizer
    Args:
      - tokenizer_path: Huggingface hub path for tokenizer
      - sample_loader: the data loader
      - data: the text dataset to train tokenizer
      - batch: the batch size for training
      - vocab_size: the vocabulary size
      - save_path: the path to save the tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.is_fast:
            logger.info(f"Use the fast version pre-trained tokenizer")
        else:
            logger.info(f"The tokenizer doesn't have fast version")
        tokenizer.train_new_from_iterator(
            sample_loader(data, batch), vocab_size=vocab_size
        )
        tokenizer.save_pretrained(save_path)
    except Exception as e:
        logger.exception(f"Finetinue Tokenizer from exists got exception: {str(e)}")


def trainBPE_Tokenizer(
    sample_loader, data, batch, vocab_size, special_tokens, save_path
):
    try:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFKD(), normalizers.StripAccents()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        if "[CLS]" not in special_tokens:
            special_tokens.append("[CLS]")
        if "[SEP]" not in special_tokens:
            special_tokens.append("[SEP]")
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens
        )
        tokenizer.train_from_iterator(sample_loader(data, batch), trainer=trainer)

        cls_token_id = tokenizer.token_to_id("[CLS]")
        sep_token_id = tokenizer.token_to_id("[SEP]")
        logger.info(f"'CLS' and 'SEP' id: {cls_token_id}, {sep_token_id}")
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_token_id),
                ("[SEP]", sep_token_id),
            ],
        )

        tokenizer.save(save_path)
    except Exception as e:
        logger.exception(f"Train a new Tokenizer got exception: {str(e)}")
