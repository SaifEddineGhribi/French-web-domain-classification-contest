# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:47:51 2020

@author: 33787
"""
import numpy as np 
from os import path
import pandas as pd
import nltk #import the natural language toolkit library
from nltk.stem.snowball import FrenchStemmer #import the French stemming library
from nltk.corpus import stopwords #import stopwords from nltk corpus
import re #import the regular expressions library; will be used to strip punctuation
from collections import Counter #allows for counting the number of occurences in a list
nltk.download('punkt')
import os #import os module
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.preprocessing import LabelEncoder


# useful functions
def read_data(train_path, test_path, texts_path):
    # Read training data
    print("reading training data ...")
    with open(train_path, 'r') as f:
        train_data = f.read().splitlines()
        train_hosts = list()
        y_train = list()
        for row in train_data:
            host, label = row.split(",")
            train_hosts.append(host)
            y_train.append(label.lower())
            
            # Read test data
    print("reading test data ...")
    with open(test_path, 'r') as f:
        test_hosts = f.read().splitlines()
    
    # Load the textual content of a set of webpages for each host into the dictionary "text". 
    # The encoding parameter is required since the majority of our text is french.
    print("reading text files ...")
    text = dict()
    filenames = os.listdir(texts_path)
    for filename in filenames:
        with open(path.join(texts_path, filename), "r", errors='ignore') as f: 
            text[filename] = f.read().replace("\n", "").lower()

    train_data = list()
    for host in train_hosts:
        if host in text:
            train_data.append(text[host])
        else:
            train_data.append('')

    test_data = list()
    for host in test_hosts:
        if host in text:
            test_data.append(text[host])
        else:
            test_data.append('')     
        
    # create dataframes
    df = pd.DataFrame(list(zip(train_data, y_train)), 
                   columns =['text', 'label']) 
    
    df_test = pd.DataFrame(list(test_data), 
                   columns =['text']) 
    
    # some preprocssing
    df['text'] = df['text'].str.replace("é", "e")
    df_test['text'] = df_test['text'].str.replace("é", "e")

    df['text'] = df['text'].str.replace("è", "e")
    df_test['text'] = df_test['text'].str.replace("è", "e")
    
    df['text'] = df['text'].str.replace("ê", "e")
    df_test['text'] = df_test['text'].str.replace("ê", "e")

    df['text'] = df['text'].str.replace("à", "a")
    df_test['text'] = df_test['text'].str.replace("à", "a")

    df['text'] = df['text'].str.replace("ù", "u")
    df_test['text'] = df_test['text'].str.replace("ù", "u")

    df['text'] = df['text'].str.replace("ô", "o")
    df_test['text'] = df_test['text'].str.replace("ô", "o")

    df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")
    df_test['text'] = df_test['text'].str.replace("[^a-zA-Z]", " ")

    df['text'] = df['text'].astype(str).str.lstrip('<').str.rstrip('+')
    df_test['text'] = df_test['text'].astype(str).str.lstrip('<').str.rstrip('+')
    print("finished")
    return df, df_test
######################################################
def get_tokens(raw,encoding='utf8'):
    '''get the nltk tokens from a text'''
    tokens = nltk.word_tokenize(raw) #tokenize the raw UTF-8 text
    return tokens
######################################################
def get_nltk_text(raw,encoding='utf8'):
    '''create an nltk text using the passed argument (raw) after filtering out the commas'''
    #turn the raw text into an nltk text object
    no_commas = re.sub(r'[.|,|\']',' ', raw) #filter out all the commas, periods, and appostrophes using regex
    tokens = nltk.word_tokenize(no_commas) #generate a list of tokens from the raw text
    text=nltk.Text(tokens,encoding) #create a nltk text from those tokens
    return text
######################################################
def get_stopswords(type="veronis"):
    '''returns the veronis stopwords in unicode, or if any other value is passed, it returns the default nltk french stopwords'''
    raw_stopword_list = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait", "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant", "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq", "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf", "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf", "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze", "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize", "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques", "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'", "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf", "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux", "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois", "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"]
    french_stops = stopwords.words('french')
    english_stops = stopwords.words('english')
    stop_words = np.concatenate([french_stops, english_stops,raw_stopword_list])
    stopword_list = [word for word in stop_words] 
    return stopword_list
######################################################   
def filter_stopwords(text,stopword_list):
    '''normalizes the words by turning them all lowercase and then filters out the stopwords'''
    words=[w.lower() for w in text] #normalize the words in the text, making them all lowercase
    #filtering stopwords
    filtered_words = [] #declare an empty list to hold our filtered words
    for word in words: #iterate over all words from the text
        if word not in stopword_list and word.isalpha() and len(word) > 1: #only add words that are not in the French stopwords list, are alphabetic, and are more than 1 character
            filtered_words.append(word) #add word to filter_words list if it meets the above conditions
    filtered_words.sort() #sort filtered_words list
    return filtered_words
######################################################   
def stem_words(words):
    '''stems the word list using the French Stemmer'''
    #stemming words
    stemmed_words = [] #declare an empty list to hold our stemmed words
    stemmer = FrenchStemmer() #create a stemmer object in the FrenchStemmer class
    for word in words:
        stemmed_word=stemmer.stem(word) #stem the word
        stemmed_words.append(stemmed_word) #add it to our stemmed word list
    stemmed_words.sort() #sort the stemmed_words
    return stemmed_words
######################################################   
def sort_dictionary(dictionary):
    '''returns a sorted dictionary (as tuples) based on the value of each key'''
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
######################################################
def normalize_counts(counts):
    total = sum(counts.values())
    return dict((word, float(count)/total) for word,count in counts.items())

######################################################        
def print_words(words):
    '''clean print the unicode words'''
    for word in words:
        print( word )
######################################################
def detokenize_words(words):
    return TreebankWordDetokenizer().detokenize(words)
#####################################################
def encode_labels(df):
    le = LabelEncoder()
    le.fit(["business/finance", "entertainment", "tech/science", "education/research", "politics/government/law",\
           "health/medical", "news/press", "sports"])
    labels = le.transform(df["label"])
    return labels
######################################################        
def pre_process_text(text, max_words = 512, do_stem = False, do_tokenize = False):    
    nltk_text = get_nltk_text(text)
    stop_words = get_stopswords()    
    filtered_words = filter_stopwords(nltk_text,stop_words)
    if do_stem :
        filtered_words = stem_words(filtered_words )
    text = text.replace("[^a-zA-Z]", " ")
    text = text.lstrip('<').rstrip('+')
    words = text.split()
    dict_words = Counter(words)
    dict_words = normalize_counts(dict_words)
    dict_words = list(sort_dictionary(dict_words)[:max_words])   
    if len(dict_words)==0:
        return -1
    x = np.array(dict_words)[:,0]
    if not do_tokenize:
        x = detokenize_words(x)
    return str(x) 
################################################################
def filter_text(text, min_word_length = 4 , min_text_length = 20, dataset = "train"):
    sentence = str(text).split(' ')    
    filtered_words = [sentence[j] for j in range(len(sentence)) if len(sentence[j])> min_word_length]
    if len(filtered_words) > min_text_length :
        text = detokenize_words(filtered_words)
    if len(filtered_words) < min_text_length and dataset == "test":
        text = sentence
    if len(filtered_words) < min_text_length and dataset == "train":
        text = "No text"  
    return text