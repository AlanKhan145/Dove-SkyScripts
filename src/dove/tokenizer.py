"""
SkyScript Tokenizer (CLIP-based BPE).

Path: src/dove/tokenizer.py

This module provides a Byte-Pair Encoding (BPE) tokenizer compatible with CLIP.
It replaces the simpler regex-based tokenizer to provide robust text encoding
for multimodal tasks.

Features:
- BPE Tokenization (matches OpenAI CLIP).
- Handles unicode characters via byte-to-unicode mapping.
- Context length padding/truncation (default 77).
- Returns PyTorch tensors.

Dependencies:
- ftfy, regex, torch
"""

import gzip
import html
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Optional

import ftfy
import regex as re
import torch

# Disable huggingface tokenizers parallelism to avoid deadlocks in dataloaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Constants & Helpers
# -----------------------------------------------------------------------------

@lru_cache()
def default_bpe() -> str:
    """Returns the path to the default BPE vocabulary file."""
    # Assumes the vocab file is located in the same directory as this script
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """Fixes text encoding and unescapes HTML entities."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Collapses multiple whitespaces into one."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# -----------------------------------------------------------------------------
# SimpleTokenizer (BPE)
# -----------------------------------------------------------------------------

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe(), special_tokens: Optional[List[str]] = None):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Load BPE merges
        try:
            with gzip.open(bpe_path, 'rb') as f:
                merges = f.read().decode("utf-8").split('\n')
        except FileNotFoundError:
            raise FileNotFoundError(f"BPE vocab file not found at: {bpe_path}. Please ensure 'bpe_simple_vocab_16e6.txt.gz' is present.")

        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        
        # Build Vocab
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
            
        # Add Special Tokens
        if not special_tokens:
            special_tokens = ['<start_of_text>', '<end_of_text>']
        else:
            special_tokens = ['<start_of_text>', '<end_of_text>'] + special_tokens
            
        vocab.extend(special_tokens)
        
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        
        # Regex pattern for tokenization
        special = "|".join(map(re.escape, special_tokens))
        self.pat = re.compile(special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


# Initialize singleton instance
# Ensure 'bpe_simple_vocab_16e6.txt.gz' is in the same folder or update default_bpe()
try:
    _tokenizer = SimpleTokenizer()
except FileNotFoundError:
    print("Warning: BPE vocab file not found. Tokenizer functions will fail until fixed.")
    _tokenizer = None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s).
    
    Args:
        texts: Input string or list of strings.
        context_length: The context length to use (CLIP default is 77).
        
    Returns:
        torch.LongTensor of shape [batch_size, context_length]
    """
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Missing vocab file?")

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.sot_token_id
    eot_token = _tokenizer.eot_token_id
    
    # Encode all texts
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # Create output tensor (init with 0, which acts as padding in this context)
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token            # Ensure EOT is present
        
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def decode(output_ids: Union[torch.Tensor, List[int]]) -> str:
    """
    Decodes a list of IDs or a Tensor back to a string.
    """
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not initialized.")

    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids.cpu().numpy().tolist()
    
    # Filter out 0 (padding) and special tokens if needed for cleaner output
    # or just raw decode:
    return _tokenizer.decode(output_ids)


__all__ = [
    "SimpleTokenizer",
    "tokenize",
    "decode",
    "default_bpe"
]