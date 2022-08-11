#!/usr/bin/python
# -*- coding: UTF-8 -*-

from enum import Enum
import numpy as np
from collections import UserDict
from typing import List, Union
from tokenization_utils import BasicTokenizer, WordpieceTokenizer, load_vocab


class PaddingStrategy(Enum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class BatchEncoding(UserDict):
    """
    Holds the output of Tokenizer
    """

    def __init__(
            self,
            data=None,
            tensor_type=None,
            prepend_batch_axis=False,
    ):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    def convert_to_tensors(self, tensor_type=None, prepend_batch_axis=False):
        """
        Convert the inner content to tensors.
        """
        if tensor_type is None:
            return self

        # Get a function reference for the correct framework
        if tensor_type == "pt":
            import torch
            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        elif tensor_type == "jt":
            import jittor
            as_tensor = jittor.Var
            is_tensor = jittor.is_var
        else:
            def _is_numpy(x):
                return isinstance(x, np.ndarray)

            as_tensor = np.asarray
            is_tensor = _is_numpy

        # Do the tensor conversion in batch
        for key, value in self.items():
            if prepend_batch_axis:
                value = [value]
            if not is_tensor(value):
                tensor = as_tensor(value)
                self[key] = tensor
        return self

    def to(self, device):
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).

        """
        self.data = {k: v.to(device=device) for k, v in self.data.items()}
        return self


class SpecialTokens:
    """
    Handle specific behaviors related to special tokens.

    Args:
        unk_token :
            A special token representing an out-of-vocabulary token.
        sep_token :
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token :
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token :
            A special token representing the class of the input (used by BERT for instance).
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
    ]

    def __init__(self, **kwargs):
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._pad_token_type_id = 0

        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if isinstance(value, str):
                    setattr(self, key, value)

    @property
    def pad_token_type_id(self):
        return self._pad_token_type_id

    @property
    def unk_token(self):
        if self._unk_token is None:
            return None
        return str(self._unk_token)

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @property
    def unk_token_id(self):
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_tokens_to_ids(value)

    @property
    def pad_token(self):
        if self._pad_token is None:
            return None
        return str(self._pad_token)

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @property
    def pad_token_id(self):
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_tokens_to_ids(value)

    @property
    def sep_token(self):
        if self._sep_token is None:
            return None
        return str(self._sep_token)

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @property
    def sep_token_id(self):
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_tokens_to_ids(value)

    @property
    def cls_token(self):
        if self._cls_token is None:
            return None
        return str(self._cls_token)

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @property
    def cls_token_id(self):
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_tokens_to_ids(value)


class BertTokenizer(SpecialTokens):
    """
    Construct a BERT tokenizer. Based on WordPiece.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
    """

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            **kwargs,
        )

        self.vocab = load_vocab(vocab_file)
        self.model_max_length = kwargs.pop("model_max_length", int(1e20))

        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def __call__(
            self,
            text,
            text_pair=None,
            add_special_tokens=True,
            padding=False,
            max_length=None,
            return_tensors=None,
            return_token_type_ids=None,
            return_attention_mask=None,
            **kwargs
    ):
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.
        """
        is_batched = isinstance(text, (list, tuple))

        if is_batched:
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self._batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
            )
        else:
            return self._encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
            )

    def _encode_plus(
            self,
            text,
            text_pair=None,
            add_special_tokens=True,
            padding=False,
            max_length=None,
            return_tensors=None,
            return_token_type_ids=None,
            return_attention_mask=None,
    ):
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.
        """

        first_ids = self.get_input_ids(text)
        second_ids = self.get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
        )

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs,
            add_special_tokens=True,
            padding=False,
            max_length=None,
            return_tensors=None,
            return_token_type_ids=None,
            return_attention_mask=None,
    ):
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
        """

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids
            else:
                ids, pair_ids = ids_or_pair_ids, None

            first_ids = self.get_input_ids(ids)
            second_ids = self.get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        return self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_tensors=return_tensors,
        )

    def prepare_for_model(
            self,
            ids,
            pair_ids=None,
            add_special_tokens=True,
            padding=False,
            max_length=None,
            return_tensors=None,
            return_token_type_ids=None,
            return_attention_mask=None,
            prepend_batch_axis=False,
    ):

        pair = bool(pair_ids is not None)

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = True
        if return_attention_mask is None:
            return_attention_mask = True

        encoded_inputs = {}

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids

        # Padding
        if padding or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding,
                return_attention_mask=return_attention_mask,
            )

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def _batch_prepare_for_model(
            self,
            batch_ids_pairs,
            add_special_tokens=True,
            padding=False,
            max_length=None,
            return_tensors=None,
            return_token_type_ids=None,
            return_attention_mask=None,
    ):

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                max_length=max_length,
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_tensors=None,  # Convert the whole batch to tensors at the end
                prepend_batch_axis=False,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def pad(
            self,
            encoded_inputs,
            padding=True,
            max_length=None,
            return_attention_mask=None,
    ):

        required_input = encoded_inputs["input_ids"]
        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, max_length = self._get_padding_strategies(
            padding=padding, max_length=max_length
        )

        if isinstance(required_input[0], (list, tuple)):  # Batch
            batch_size = len(required_input)

            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = max(len(inputs) for inputs in required_input)
                padding_strategy = PaddingStrategy.MAX_LENGTH

            batch_outputs = {}
            for i in range(batch_size):
                inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
                outputs = self._pad(
                    inputs,
                    max_length=max_length,
                    padding_strategy=padding_strategy,
                    return_attention_mask=return_attention_mask,
                )

                for key, value in outputs.items():
                    if key not in batch_outputs:
                        batch_outputs[key] = []
                    batch_outputs[key].append(value)

            return BatchEncoding(batch_outputs)
        else:  # Not Batch
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs)

    def _pad(
            self,
            encoded_inputs,
            max_length=None,
            padding_strategy=PaddingStrategy.DO_NOT_PAD,
            return_attention_mask=None,
    ):
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = True

        required_input = encoded_inputs["input_ids"]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if return_attention_mask:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                )
            encoded_inputs["input_ids"] = required_input + [self.pad_token_id] * difference

        return encoded_inputs

    def _get_padding_strategies(self, padding=False, max_length=None):
        """
        Find the correct padding strategy
        """
        # Get padding strategy
        padding_strategy = PaddingStrategy.DO_NOT_PAD
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding

        # Set max length if needed
        if max_length is None:
            max_length = self.model_max_length

        return padding_strategy, max_length

    def get_input_ids(self, text):
        text = text[:self.model_max_length]
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            split_tokens += self.wordpiece_tokenizer.tokenize(token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
