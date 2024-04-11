import regex as re
from collections import Counter
import os
import sys
import pickle
import gc
import json
import tqdm

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        if special_tokens != None:
            self.regex_pattern = r"""'(?:[sdmt]|ll|ve|re)|""" + rf"|".join(["(?:" + re.escape(sp) + ")" for sp in self.special_tokens])  + r"""| ?\p{L}+| ?\p{N}+|""" + rf"|".join([" ?[^\s\p{L}\p{N}]+ *(?=" + re.escape(sp) + ")" for sp in self.special_tokens]) + r"""| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        else:
            self.regex_pattern = r""" *<\|endoftext\|>|'(?:[sdmt]|ll|ve|re)| \p{L}+| \p{N}+|(?:(?<!<)endoftext(?!>))|?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        return
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as f:
            vocab_list = json.load(f)
        with open(merges_filepath, 'r') as f:
            merges_list = f.read()
        return BPETokenizer(vocab_list, merges_list, special_token=special_tokens)
    
    def encode(self, text):
        PAT = self.regex_pattern
        pretokenized = re.findall(PAT, text)
        pretokenized_encoded = []
        invert_vocab = {v : k for k, v in self.vocab.items()}
        
        for w in pretokenized:
            if w not in self.special_tokens:
                encoded = [invert_vocab[char.encode('utf-8')] for char in w]
            else:
                encoded = [invert_vocab[w.encode("utf-8")]]
            pretokenized_encoded.append(encoded)
        
        unique_tokens = list(set(pretokenized_encoded))
        pairs_to_tokens = self.get_pairs(unique_tokens)
        token_splits = dict(zip(unique_tokens, unique_tokens))

        for pair_bytes in self.merges:
            pair_indices = (invert_vocab[pair_bytes[0]], invert_vocab[pair_bytes[1]])
            new_merge = invert_vocab[pair_bytes[0] + pair_bytes[1]]
            
            if pair_indices in pairs_to_tokens:
                for tok in pairs_to_tokens[pair_indices]:
                    t = token_splits[tok]
                    new_token = []
                    index_pairs = [(t[i], t[i + 1]) for i in range(len(t) - 1)]
                    k = 0
                    n_merges = 0
                    while k < len(index_pairs):
                        pair = index_pairs[k]
                        if pair == pair_indices:
                            new_token.append(new_merge)
                            if k + 1 == len(index_pairs) - 1:
                                new_token.append(index_pairs[k+1][1])
                            k += 1
                            n_merges += 1
                        elif k == len(index_pairs) - 1:
                            new_token.append(pair[0])
                            new_token.append(pair[1])
                        else:
                            new_token.append(pair[0])
                        k += 1
                    new_token = tuple(new_token)
                    token_splits[t] = new_token
                    new_pairs = [(new_token[i], new_token[i + 1]) for i in range(len(new_token) - 1)]
                    
                    #Adding new pairs
                    for p in set(new_pairs) - set(index_pairs):
                        if p not in pairs_to_tokens:
                            pairs_to_tokens[p] = {tok}
                        else:
                            pairs_to_tokens[p] = pairs_to_tokens[p].union({tok})

                    # Removing old pairs
                    for p in set(index_pairs) - set(new_pairs):
                        pairs_to_tokens[p] -= {tok}
            
            tokenized = list()
            for w in pretokenized_encoded:
                tokenized += token_splits[w]

        return tokenized 
    
    def get_pairs(pretokenized_encoded):
        pair_to_index = dict()

        for encoding in enumerate(tqdm(pretokenized_encoded)):
            index_pairs = [(encoding[j], encoding[j + 1]) for j in range(len(encoding) - 1)]
            for pair in index_pairs:
                if pair in pair_to_index:
                    pair_to_index[pair] = pair_to_index.union({encoding})
                else:
                    pair_to_index[pair] = {encoding}
        return pair_to_index

    def encode_iterable(self, iterable):

        return
    
    def decode(self, ids):
        return