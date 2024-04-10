import regex as re
from collections import Counter
import os
import sys
import pickle


def train_tokenizer(input_path, vocab_size, special_tokens):
    merges = []
    vocab = {x : bytes([x]) for x in range(256)}
    for x in range(len(special_tokens)):
        vocab[256+x] = special_tokens[x].encode('utf-8')
    PAT = r""" *<\|endoftext\|>|'(?:[sdmt]|ll|ve|re)|\s*?\p{L}+|\s*?\p{N}+|(?:(?<!<)endoftext(?!>))|\s*?[^\s\p{L}\p{N}<]+"""
    with open(input_path, 'r') as f:
        corpus = f.read()
    pretokenized = re.findall(PAT, corpus)
    pretokenized = [tuple(w.encode('utf-8')) if "<|endoftext|>" not in w else tuple([256]) if w == "<|endoftext|>" else tuple(w.split('<|endoftext|>')[0].encode('utf-8')) + tuple([256]) for w in pretokenized]
    pretokenized_counter = Counter(pretokenized)
    pairs_counter, pairs_to_tokens = get_pairs(pretokenized_counter)
    token_splits = list(pretokenized_counter.keys())

    i= 256+len(special_tokens)
    while i < vocab_size:
        new_merge = i
        max_value = max(pairs_counter.values())
        merge_vocab = max([key for key, val in pairs_counter.items() if val == max_value])
        merges.append((vocab[merge_vocab[0]], vocab[merge_vocab[1]])) #these are probably indices but we want bytes?
        vocab[new_merge] = vocab[merge_vocab[0]] + vocab[merge_vocab[1]] #these are also indices right now
        indices = list(pairs_to_tokens[merge_vocab]) #this indexes the pretokenized_counter
        matching_tokens = dict(zip([token_splits[i] for i in indices], [list(pretokenized_counter.values())[i] for i in indices]))
        new_tokens = dict()
        
        for j, (t, c) in enumerate(matching_tokens.items()):
            new_token = []
            index_pairs = [(t[i], t[i + 1]) for i in range(len(t) - 1)]
            for k, pair in enumerate(index_pairs):
                if pair == merge_vocab:
                    new_token.append(new_merge)
                elif k == len(index_pairs)-1:
                    new_token.append(pair[0])
                    new_token.append(pair[1])
                else:
                    new_token.append(pair[0])
            new_token = tuple(new_token)
            new_tokens[new_token] = c
            token_splits[indices[j]] = new_token

        remove_count, remove_indices = get_pairs(matching_tokens)
        add_count, add_indices  = get_pairs(new_tokens)
        for j in range(len(remove_count)):
            pairs_counter[list(remove_count.keys())[j]] -= list(remove_count.values())[j]
            pairs_to_tokens[list(remove_indices.keys())[j]] -= {indices[x] for x in list(remove_indices.values())[j]}
        for j in range(len(add_count)):
            if list(add_count.keys())[j] not in pairs_counter.keys():
                pairs_counter[list(add_count.keys())[j]] = list(add_count.values())[j]
                pairs_to_tokens[list(add_indices.keys())[j]] = {indices[x] for x in list(add_indices.values())[j]}
            else:
                pairs_counter[list(add_count.keys())[j]] += list(add_count.values())[j]
                pairs_to_tokens[list(add_indices.keys())[j]].union({indices[x] for x in list(add_indices.values())[j]})
        
        del pairs_counter[merge_vocab]
        del pairs_to_tokens[merge_vocab]
        i += 1

    return vocab, merges

def get_pairs(pretokenized_counts):
    pair_counter = Counter()
    pair_to_index = dict()

    for i, (index, count) in enumerate(pretokenized_counts.items()):
        if len(index) == 1:
            continue
        index_pairs = [(index[i], index[i + 1]) for i in range(len(index) - 1)]
        for pair in index_pairs:
            pair_counter[pair] += count
            if pair in pair_to_index:
                pair_to_index[pair] = pair_to_index[pair].union({i})
            else:
                pair_to_index[pair] = {i}
    return pair_counter, pair_to_index


if __name__=="__main__":
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    vocab_size = int(sys.argv[3])
    dataset = sys.argv[4]
    special_tokens = ['<|endoftext|>']
    vocab, merges = train_tokenizer(file_path, vocab_size, special_tokens)
    with open(out_path + dataset + '.vocab') as f:
        pickle.dump(vocab, f)
    with open(out_path + dataset + '.merges') as f:
        pickle.dump(merges, f)