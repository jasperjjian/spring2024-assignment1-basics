import regex as re
from collections import Counter
import os
import sys
import pickle
import gc
import json
from tqdm import tqdm


def train_tokenizer(input_path, vocab_size, special_tokens):
    merges = []
    vocab = {x : bytes([x]) for x in range(256)}
    for x in range(len(special_tokens)):
        vocab[256+x] = special_tokens[x].encode('utf-8')
    #PAT = r""" *<\|endoftext\|>|'(?:[sdmt]|ll|ve|re)| \p{L}+| \p{N}+|(?:(?<!<)endoftext(?!>))|?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)|(?:<\|endoftext\|>)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+ *(?=<\|endoftext\|>)| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    print("Loading corpus: ")
    with open(input_path, 'r') as f:
        corpus = f.readlines()
    special_tokens_regex = [re.findall(PAT, spec) for spec in special_tokens]
    longest_token = max([len(sp) for sp in special_tokens_regex])
    pretokenized_counter = Counter()
    print("Pretokenizing: ")
    for l in tqdm(corpus):
        pretokenized = re.findall(PAT, l)
        for w in pretokenized:
            if "<|endoftext|>" not in w:
                encoded = w.encode('utf-8')
            else:
                encoded = tuple([256])
            pretokenized_counter[encoded] += 1
    del corpus
    gc.collect()
    #print("Counting")
    del pretokenized
    gc.collect()
    """pretokenized = [tuple(w.encode('utf-8')) if "<|endoftext|>" not in w else tuple([256]) for w in pretokenized]
    pretokenized_counter = Counter(pretokenized)"""
    pairs_counter, pairs_to_tokens = get_pairs(pretokenized_counter)
    token_splits = dict(zip(pretokenized_counter.keys(), pretokenized_counter.keys()))
    print("Running stuff")
    i= 256+len(special_tokens)
    progress_bar = tqdm(total=vocab_size-i)
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
        progress_bar.update(1)
    progress_bar.close()
    return vocab, merges

def get_second_tuple_value(item):
    return item[1]

def get_pairs(pretokenized_counts):
    pair_counter = Counter()
    pair_to_index = dict()
    print("Counting Pairs")
    for k, (index, count) in enumerate(tqdm(pretokenized_counts.items())):
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
        pickle.dump(vocab, f)
    with open(out_path + dataset + '.merges.txt', 'w') as f:
        for byte_tuple in merges:
            string_tuple = tuple(str(byte) for byte in byte_tuple)
            f.write(','.join(string_tuple) + '\n')

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        if special_tokens != None:
            self.regex_pattern = r"""'(?:[sdmt]|ll|ve|re)|""" + rf"|".join(["(?:" + re.escape(sp) + ")" for sp in self.special_tokens])  + r"""| ?\p{L}+| ?\p{N}+|""" + rf"|".join([" ?[^\s\p{L}\p{N}]+ *(?=" + re.escape(sp) + ")" for sp in self.special_tokens]) + r"""| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        else:
            self.regex_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        return
    
    @classmethod
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
            if self.special_tokens == None:
                encoded = (invert_vocab[bytes([char])] for char in list(w.encode('utf-8')))
            elif w not in self.special_tokens:
                encoded = (invert_vocab[char.encode('utf-8')] for char in w)
            else:
                encoded = (invert_vocab[w.encode("utf-8")])
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
        print(tokenized)
        return tokenized 
    
    def get_pairs(self, pretokenized_encoded):
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

        raise NotImplementedError
    
    def decode(self, ids):
        decoded_bytes = bytes()
        for x in ids:
            decoded_bytes += self.vocab[x]

        return decoded_bytes.decode("utf-8", errors='replace')