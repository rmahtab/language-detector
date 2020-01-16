import sys
import io
import operator
from scipy.stats import pareto
import matplotlib.pyplot as plt
import numpy as np
import math
import string


#### Read in Data

with io.open("/Users/shahabmahtab/CMU/Junior/LangStat/HW5/data/1.txt", "r", encoding = "utf-8") as my_file:
    body = my_file.read()
word_list = body.split()
num_tokens = len(word_list)

#### Unigrams

unigram_freq = dict()

for word in word_list:
    if word in unigram_freq.keys():
        unigram_freq[word] += 1
    else:
        unigram_freq[word] = 1

sorted_unigram_freq = sorted(unigram_freq.items(), key = operator.itemgetter(1))
n1 = len(sorted_unigram_freq)

#### Bigrams

bigram_freq = dict()

for i in range(len(word_list)-1):
    curr_bigram = word_list[i] + " " + word_list[i+1]
    if curr_bigram in bigram_freq.keys():
        bigram_freq[curr_bigram] += 1
    else:
        bigram_freq[curr_bigram] = 1

sorted_bigram_freq = sorted(bigram_freq.items(), key = operator.itemgetter(1))
n2 = len(sorted_bigram_freq)

#### Trigrams

trigram_freq = dict()

for i in range(len(word_list)-2):
    curr_trigram = word_list[i] + " " + word_list[i+1] + " " + word_list[i+2]
    if curr_trigram in trigram_freq.keys():
        trigram_freq[curr_trigram] += 1
    else:
        trigram_freq[curr_trigram] = 1

sorted_trigram_freq = sorted(trigram_freq.items(), key = operator.itemgetter(1))
n3 = len(sorted_trigram_freq)

#### Symbol Frequencies

symbol_counts = dict()
num_chars = len(body)

for c in range(num_chars):
    if body[c] in symbol_counts.keys():
        symbol_counts[body[c]] += 1
    else:
        symbol_counts[body[c]] = 1

symbol_freqs = dict()

for s in symbol_counts.keys():
    symbol_freqs[s] = symbol_counts[s] / (num_chars)

#### Symbol Pairs

symbol_rules = dict()

for q in range(num_chars-1):
    if body[q] in symbol_rules.keys():
        symbol_rules[body[q]].add(body[q+1])
    else:
        symbol_rules[body[q]] = set()
        symbol_rules[body[q]].add(body[q+1])

#### Character Entropy

max_entropy_freqs = [1/len(symbol_counts)]*len(symbol_counts)
freq_array = []

for n in symbol_freqs.values():
    freq_array.append(n)

def entropy(px):
    sum = 0
    
    for x_i in range(len(px)):
        sum += px[x_i] * math.log(px[x_i], 2)
    
    return (-1)*sum

#### Unigram KL Divergence from Pareto

uni_ranks = np.arange(2, n1+1)
uni_freqs = []
for u in range(n1-2, -1, -1):
    uni_freqs.append(sorted_unigram_freq[u][1])
uni_freqs = np.array(uni_freqs)

uni_beta = np.amin(uni_ranks)
uni_alpha = n1 / (np.sum(np.log(uni_ranks)) - n1*math.log(uni_beta))



def pareto(x, a, b):
    return (a*(b**a))/(x**(a+1))

'''
plt.figure("Unigram_rmahtab.pdf")
plt.plot(np.log(uni_ranks), np.log(uni_freqs), 'ro')
plt.plot(np.log(uni_ranks), np.log(pareto(uni_ranks, uni_alpha, uni_beta)))
plt.xlabel("log(rank)")
plt.ylabel("log(frequency)")
plt.show()
'''

def kl_divergence(p, q):
    sum = 0
    
    for i in range(p.shape[0]):
        if p[i] != 0 and q[i] != 0:
            sum += p[i] * math.log(p[i]/abs(q[i]))
    
    return abs(sum)

#### Print Stats

print("Number of word-tokens: " + str(num_tokens))
print("Number of unigrams/word-types: " + str(n1))
print("Number of bigrams: " + str(n2))
print("Number of trigrams: " + str(n3))
#print(sorted_unigram_freq)

print("\n")

print("Symbol counts: " + str(symbol_counts))
print("Number of unique symbols (including whitespaces): " + str(len(symbol_counts)))
print("Number of total symbols: " + str(num_chars))

print("\n")

#print("Symbol frequencies (not including spaces): " + str(symbol_freqs))
print("Entropy of symbols: " + str(entropy(freq_array)))
print("Max entropy of symbols: " + str(entropy(max_entropy_freqs)))
print("Relative entropy of symbols: " + str(entropy(freq_array)/entropy(max_entropy_freqs)))
print("Unigram KL Divergence: " + str(kl_divergence(np.log(uni_freqs), np.log(pareto(uni_ranks, uni_alpha, uni_beta)))))