import regex as re
from collections import Counter
import os
import sys
import pickle
import gc
import json


def train_tokenizer(input_path, vocab_size, special_tokens):
    merges = []
    vocab = {x : bytes([x]) for x in range(256)}
    for x in range(len(special_tokens)):
        vocab[256+x] = special_tokens[x].encode('utf-8')
    #PAT = r""" *<\|endoftext\|>|'(?:[sdmt]|ll|ve|re)| \p{L}+| \p{N}+|(?:(?<!<)endoftext(?!>))|?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)|(?:<\|endoftext\|>)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+ *(?=<\|endoftext\|>)| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    print("Loading corpus: ")
    with open(input_path, 'r') as f:
        corpus = f.read()
    special_tokens_regex = [re.findall(PAT, spec) for spec in special_tokens]
    longest_token = max([len(sp) for sp in special_tokens_regex])
    print("Pretokenizing: ")
    pretokenized = re.findall(PAT, corpus)
    del corpus
    gc.collect()
    print("Counting")
    pretokenized_counter = Counter()
    for w in pretokenized:
        if "<|endoftext|>" not in w:
            encoded = w.encode('utf-8')
        else:
            encoded = tuple([256])
        pretokenized_counter[encoded] += 1
    del pretokenized
    gc.collect()
    """pretokenized = [tuple(w.encode('utf-8')) if "<|endoftext|>" not in w else tuple([256]) for w in pretokenized]
    pretokenized_counter = Counter(pretokenized)"""
    pairs_counter, pairs_to_tokens = get_pairs(pretokenized_counter)
    token_splits = dict(zip(pretokenized_counter.keys(), pretokenized_counter.keys()))
    print("Running stuff")
    i= 256+len(special_tokens)
    while i < vocab_size:
        new_merge = i
        max_value = max(pairs_counter.values())
        merge_vocab, merge_bytes = max([((key[0], key[1]), (vocab[key[0]],vocab[key[1]])) for key, val in pairs_counter.items() if val == max_value], key=get_second_tuple_value)
        merges.append(merge_bytes) 
        vocab[new_merge] = vocab[merge_vocab[0]] + vocab[merge_vocab[1]] 
        indices = list(pairs_to_tokens[merge_vocab]) 
        matching_tokens = zip([token_splits[i] for i in indices], [pretokenized_counter[i] for i in indices])

        for j, (t, c) in zip(indices, matching_tokens):
            new_token = []
            index_pairs = [(t[i], t[i + 1]) for i in range(len(t) - 1)]
            original_counts = Counter(index_pairs)
            new_pairs = []
            k = 0
            n_merges = 0
            while k < len(index_pairs):
                pair = index_pairs[k]
                if pair == merge_vocab:
                    new_token.append(new_merge)
                    if k - 1 >= 0:
                        new_pairs.append((index_pairs[k-1][0], new_merge))
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
            original_pair_set = set(index_pairs)
            new_pairs1 = [(new_token[i], new_token[i + 1]) for i in range(len(new_token) - 1)]
            new_counts = Counter(new_pairs1)
            for p, v in (original_counts - new_counts).items():
                pairs_counter[p] -= v*c
                if new_counts[p] == 0:
                    pairs_to_tokens[p] -= {j}
            for p, v in (new_counts - original_counts).items():
                pairs_counter[p] += v*c
                if p not in pairs_to_tokens:
                    pairs_to_tokens[p] = {j}
                else:
                    pairs_to_tokens[p] = pairs_to_tokens[p].union({j})
            token_splits[j] = new_token
            assert len(index_pairs) - n_merges == len(new_pairs1)
        i += 1
        del pairs_counter[merge_vocab]
        if i % 1000 == 0:
            print(merges)
    return vocab, merges

def get_second_tuple_value(item):
    return item[1]

def get_pairs(pretokenized_counts):
    pair_counter = Counter()
    pair_to_index = dict()

    for k, (index, count) in enumerate(pretokenized_counts.items()):
        index_pairs = [(index[i], index[i + 1]) for i in range(len(index) - 1)]
        for pair in index_pairs:
            pair_counter[pair] += count
            if pair in pair_to_index:
                pair_to_index[pair] = pair_to_index[pair].union({index})
            else:
                pair_to_index[pair] = {index}
    return pair_counter, pair_to_index


if __name__=="__main__":
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    vocab_size = int(sys.argv[3])
    dataset = sys.argv[4]
    special_tokens = ['<|endoftext|>']
    vocab, merges = train_tokenizer(file_path, vocab_size, special_tokens)
    #print(vocab)
    #print(merges)
    with open(out_path + dataset + '.vocab', 'wb') as f:
        json.dump(vocab, f)
    with open(out_path + dataset + '.merges', 'w') as f:
        for byte_tuple in merges:
            string_tuple = tuple(str(byte) for byte in byte_tuple)
            f.write(','.join(string_tuple) + '\n')