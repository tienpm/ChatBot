import re

import tokenizers
from transformers import AutoTokenizer
from underthesea import word_tokenize


class TokenizerMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __keep_text_only(self, text):
        """
        Normalize text and keep characters and numbers only
        Args:
          - text: string, input text to remove special characters
        Returns:
          - cleaned_text: string, cleaned text
        """
        cleaned_text = re.sub(r"\d+|[^\w\s]", "", text)
        return cleaned_text

    def get_compress_ratio(self, text):
        text = self.__keep_text_only(text)
        raw_size = len(text.encode("utf-8"))

        encoded_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if isinstance(encoded_ids, tokenizers.Encoding):
            encoded_ids = encoded_ids.ids
        print(type(encoded_ids))
        print(encoded_ids)
        # For through encode id then decode and encode to utf-8
        tokens_size = [
            len(self.tokenizer.decode([token_id]).encode("utf-8"))
            for token_id in encoded_ids
        ]
        tokenized_size = sum(tokens_size)
        print(raw_size, tokenized_size)

        compression_rate = raw_size / tokenized_size

        return compression_rate

    def get_tokens_per_byte(self, text):
        text = self.__keep_text_only(text)
        tokens_encode_id = self.tokenizer.encode(text, add_special_tokens=False)
        if isinstance(tokens_encode_id, tokenizers.Encoding):
            tokens_encode_id = tokens_encode_id.ids
        tokens = [self.tokenizer.decode([token_id]) for token_id in tokens_encode_id]
        tokens_per_byte = len(tokens) / len(text.encode("utf-8"))

        return tokens_per_byte

    def get_subword_fertility(self, text):
        """
        Calculates the average subword fertility, which calculates the average number of subwords produced per tokenized word

        Args:
            tokenizer: The tokenizer to evaluate (e.g., from transformers).
            texts: A list of text strings to tokenize.

        Returns:
            The average subword fertility (float).
        """
        text = self.__keep_text_only(text)
        total_subwords = 0

        words = word_tokenize(text)
        n_words = len(words)

        # tokenize_res = []
        for word in words:
            tokens = self.tokenizer.encode(word)
            total_subwords += len(tokens)
            # print(word, len(tokens))

        # print(tokenize_res)
        average_fertility = total_subwords / n_words if n_words > 0 else 0

        return average_fertility

    def get_proportion_continued_words(self, text):
        """
          Calculates the proportion of continued words (split into multiple subwords) for a tokenizer on a set of texts.

        Args:
            tokenizer: The tokenizer to evaluate (e.g., from transformers).
            texts: A list of text strings to tokenize.

        Returns:
            The proportion of continued words (float).
        """
        text = self.__keep_text_only(text)
        total_words = 0
        continued_words = 0

        words = word_tokenize(text)
        total_words = len(words)

        for word in words:
            tokens = self.tokenizer.encode(word)
            continued_words += len(tokens)  # Count subwords with continuation marker

        pcw = continued_words / total_words if total_words > 0 else 0
        return pcw

        def get_coverage_metrics(self, text):
            """
            Calculates the proportion of unknown words or rarely used tokens in a tokenized corpus

            Args:
                tokenizer: The tokenizer to evaluate (e.g., from transformers).
                texts: A list of text strings to tokenize.
                rare_threshold: The frequency below which a token is considered rare.

            Returns:
                A dictionary containing:
                    - "oov_rate": Proportion of unknown words (float).
                    - "rare_token_proportion": Proportion of rare tokens (float).
            """
            raise NotImplementedError


if __name__ == "__main__":
    test_texts = [
        "Internet Society hay ISOC là một tổ chức quốc tế hoạt động phi lợi nhuận, phi chính phủ và bao gồm các thành viên có trình độ chuyên ngành",
        "The lower the fertility, the better the tokenizer is at preserving word boundaries.",
        "tổ chức thành viên",
    ]
    evaluator = TokenizerMetrics("meta-llama/Llama-2-7b-chat-hf")
    for text in test_texts:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        print(f"Tokenizer output: {tokenizer.tokenize(text)}")
        print(f"Compression ratio: {evaluator.get_compress_ratio(text)}")
        print(f"Tokens per byte v1: {evaluator.get_tokens_per_byte(text)}")
        print(f"Average subword fertility: {evaluator.get_subword_fertility(text)}")
        print(
            f"Proportion of continued words: {evaluator.get_proportion_continued_words(text)}"
        )
        print("==================================================")
