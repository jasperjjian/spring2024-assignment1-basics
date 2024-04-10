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
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, 'r') as f:
        corpus = f.read()
    pretokenized = re.findall(PAT, corpus)
    #print(pretokenized[:500])
    pretokenized = [tuple(w.encode('utf-8')) if "<|endoftext|>" not in w else tuple([256]) if w == "<|endoftext|>" else tuple(w.split('<|endoftext|>')[0].encode('utf-8')) + tuple([256]) for w in pretokenized]
    pretokenized_counter = Counter(pretokenized)
    pairs_counter, pairs_to_tokens = get_pairs(pretokenized_counter)
    token_splits = list(pretokenized_counter.keys())

    i= 256+len(special_tokens)
    while i < vocab_size:
        #print(i-256)
        new_merge = i
        max_value = max(pairs_counter.values())
        #print([((key[0], key[1]), (vocab[key[0]],vocab[key[1]]), val) for key, val in pairs_counter.items() if val == max_value])
        #print((b' th', b'at') in [(vocab[key[0]],vocab[key[1]]) for key in pairs_counter.keys()])
        merge_vocab, merge_bytes = max([((key[0], key[1]), (vocab[key[0]],vocab[key[1]])) for key, val in pairs_counter.items() if val == max_value], key=get_second_tuple_value)
        merges.append((merge_bytes[0], merge_bytes[1])) #these are probably indices but we want bytes?
        vocab[new_merge] = vocab[merge_vocab[0]] + vocab[merge_vocab[1]] #these are also indices right now
        indices = list(pairs_to_tokens[merge_vocab]) #this indexes the pretokenized_counter
        matching_tokens = dict(zip([token_splits[i] for i in indices], [list(pretokenized_counter.values())[i] for i in indices]))

        for j, (t, c) in enumerate(matching_tokens.items()):
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
                        #pairs_counter[index_pairs[k-1]] -= c
                        #new_pairs = new_pairs[:-1]
                        new_pairs.append((index_pairs[k-1][0], new_merge))
                        #pairs_counter[(index_pairs[k-1][0], new_merge)] += c
                    if k + 1 < len(index_pairs):
                        #pairs_counter[index_pairs[k+1]] -= c
                        #new_pairs.append((new_merge, index_pairs[k+1][1]))
                        #pairs_counter[(new_merge, index_pairs[k+1][1])] += c
                        if k + 1 == len(index_pairs) - 1:
                            new_token.append(index_pairs[k+1][1])
                    k += 1
                    n_merges += 1
                elif k == len(index_pairs) - 1:
                    new_token.append(pair[0])
                    new_token.append(pair[1])
                    #new_pairs.append(pair)
                else:
                    new_token.append(pair[0])
                    #new_pairs.append(pair)
                k += 1
            new_token = tuple(new_token)
            original_pair_set = set(index_pairs)
            new_pairs1 = [(new_token[i], new_token[i + 1]) for i in range(len(new_token) - 1)]
            new_counts = Counter(new_pairs1)
            for p, v in (original_counts - new_counts).items():
                pairs_counter[p] -= v*c
            for p, v in (new_counts - original_counts).items():
                pairs_counter[p] += v*c
            """print(new_pairs)
            print(new_pairs1)
            print(merge_vocab)
            print(t)    
            print(new_token)"""
            #assert new_pairs1 == new_pairs
            new_pair_set = set(new_pairs1)
            removed_pairs = original_pair_set - new_pair_set
            added_pairs = new_pair_set - original_pair_set
            for p in removed_pairs:
                pairs_to_tokens[p] -= {indices[j]}
            for p in added_pairs:
                if p not in pairs_to_tokens:
                    pairs_to_tokens[p] = {indices[j]}
                else:
                    pairs_to_tokens[p].union({indices[j]})
            token_splits[indices[j]] = new_token
            assert len(t) - n_merges == len(new_token)
        
        i += 1
        del pairs_counter[merge_vocab]

    return vocab, merges

def get_second_tuple_value(item):
    return item[1]

def get_pairs(pretokenized_counts):
    pair_counter = Counter()
    pair_to_index = dict()

    for i, (index, count) in enumerate(pretokenized_counts.items()):
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
    #print(vocab)
    print(merges)
    """with open(out_path + dataset + '.vocab') as f:
        pickle.dump(vocab, f)
    with open(out_path + dataset + '.merges') as f:
        pickle.dump(merges, f)"""