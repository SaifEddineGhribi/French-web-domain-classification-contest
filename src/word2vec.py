# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:19:24 2020

@author: 33787
"""
import numpy as np 
import gzip

class Word2Vec():

    def __init__(self, filepath, vocab_size=50000):
        self.words, self.embeddings = self.load_wordvec(filepath, vocab_size)
        # Mappings for O(1) retrieval:
        self.word2id = {word: idx for idx, word in enumerate(self.words)}
        self.id2word = {idx: word for idx, word in enumerate(self.words)}
    
    def load_wordvec(self, filepath, vocab_size):
        assert str(filepath).endswith('.gz')
        words = []
        embeddings = []
        with gzip.open(filepath, 'rt', encoding ="utf8") as f:  # Read compressed file directly
            next(f)  # Skip header
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                words.append(word)
                embeddings.append(np.fromstring(vec, sep=' '))
                if i == (vocab_size - 1):
                    break
        print('Loaded %s pretrained word vectors' % (len(words)))
        return words, np.vstack(embeddings)
    
    def encode(self, word):
        # Returns the 1D embedding of a given word
        if word in self.words:
            return  self.embeddings[self.words.index(word)]
        else:
            return 0
    
    def score(self, word1, word2):
        # Return the cosine similarity: use np.dot & np.linalg.norm
        emb1 = self.encode(word1)
        emb2 = self.encode(word2)
        if (len(emb1)==0) or (len(emb2)==0):
            return 0
        else:
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 
    
    def most_similar(self, word, k=5):
        # Returns the k most similar words: self.score & np.argsort 
        if (word not in self.words):
            print('input word is not in word2vec loaded vocab')
            return 0
        else:
            similarity_scores = [self.score(word, w_dict) for w_dict in self.words]
            idx_matches = np.argsort(similarity_scores)[::-1][:k]
            matches = [self.words[i] for i in idx_matches]
        return matches
################################################################################################
        
class BagOfWords():
    
    def __init__(self, word2vec):
        self.word2vec = word2vec
    
    def build_idf(self, sentences):
        # build the idf dictionary: associate each word to its idf value
        # -> idf = {word: idf_value, ...}
        idf = {}
        for sent in sentences:
            for w in set(sent.split()):
                idf[w] = idf.get(w, 0) + 1        
        for w in idf.keys():
            idf[w] = max(1, np.log10(len(sentences) / (idf[w])))
        return idf
    
    def encode(self, sentence, idf=None):
        # Takes a sentence as input, returns the sentence embedding
        idf_dic = idf
        somme = np.zeros(300)
        count = 0
        for word in sentence.split() :
            if word in self.word2vec.words :
                if idf is None:
                    somme += np.array(list(self.word2vec.encode(word)))
                    count +=1
                else : 
                    somme += idf_dic[word] * np.array(list(self.word2vec.encode(word)))
                    count +=1
        if count==0: #if any word in the sentence is in our lookup table
            return np.zeros(300)
        else:
            return somme/count

    def score(self, sentence1, sentence2, idf=None):
        # cosine similarity: use np.dot & np.linalg.norm 
        emb1 = self.encode(sentence1, idf)
        emb2 = self.encode(sentence2, idf)
        # edge case: 0-vector embedding
        if len(emb1)==0 or len(emb2)==0:
            return 0
        else:
            similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity
    
    def most_similar(self, sentence, sentences, idf=None, k=5):
        # Return most similar sentences
        scores = [self.score(sentence, s2, idf) for s2 in sentences]
        ids = np.flip(np.argsort(scores))
        similar_sentences = [sentences[id] for id in ids[1:k+1]]
        return similar_sentences