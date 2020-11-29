import numpy as np
import json
import string
from string import punctuation
import nltk
from nltk.corpus import stopwords
import sys
import os


stopwords = stopwords.words('english')
def leaves(tree):
    '''
    Finds NP (nounphrase) leaf nodes of a chunk tree.
    '''
    l = []
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        l.append(subtree.leaves())
    
    return l

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    #word = stemmer.stem(word)
    #word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    '''
    Checks conditions for acceptable word: length, stopword.
    '''
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    kp = []
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        if term:
            kp.append(term)
    
    return kp

def get_kp(text):
    ''' 
    get kps from a document
    Inputs:
        document : a word list : ['sun', 'sunshine', ...] || lower cased
        kps : can have more than one kp : [['sun'], ['key','phrase'], ['sunshine']] || not duplicate
    Outputs:
        all_present_kps : present keyphrases
        positions_for_all : start_end_posisiton for prensent keyphrases
        a present kp postions list : every present's positions in documents, 
        each kp can be presented in several postions .
        [[[0,0],[20,21]], [[1,1]]]
    '''
    # Used when tokenizing words
    sentence_re = r'''(?x)      # set flag to allow verbose regexps
            (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
          | \w+(?:-\w+)*        # words with optional internal hyphens
          | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
          | \.\.\.              # ellipsis
          | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        '''

    #lemmatizer = nltk.WordNetLemmatizer()
    #stemmer = nltk.stem.porter.PorterStemmer()

    #Taken from Su Nam Kim Paper
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
            {<JJ|JJR|JJS|VBG|VBN>*<NN|NNS|NNP|NNPS|VBG>}
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """

    #toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(text)
    chunker = nltk.RegexpParser(grammar)
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    #pos,pos_set = find_positions(text,terms)
    return terms#,pos,pos_set

def find_positions(document,bert_tocs, kps):
    ''' 
    merge the same kps & keep present kps in document
    Inputs:
        document : a word list lower cases : ['sun', 'sunshine', ...]
        bert_tocs: bert tokenized word list in lower case : ['try', 'ing', 'to', .....]
        kps : can have more than one kp no duplicates : [['sun'], ['key','phrase'], ['sunshine']]
    Outputs:
        pos_list : list of start anf end positionas of all matching KPs : [[1,4,9,....],[2,6,10,...]]
        pos_set : set of start and end position tuples : [(1,2),(4,6),(9,10),.....]
    '''
    tot_doc_char = ' '.join(document)
    
    positions_for_all = []
    position_start,position_end =[],[]
    all_present_kps = []
    for kp in kps:
        if len(kp)<1:
            continue
        ans_string = ' '.join(kp)
        
        if ans_string not in tot_doc_char:
            continue
        else: 
            positions_for_each = []
            # find all positions for each kp
            for i in range(0, len(bert_tocs) - len(kp) + 1):
                found = False
                search_str = ''
                if ans_string.startswith(bert_tocs[i]):
                    found = True
                    search_str +=bert_tocs[i]
                    search_idx = i
                    while found and search_idx<(len(bert_tocs)-1):
                        search_idx+=1
                        if search_str+bert_tocs[search_idx] in ans_string:
                            search_str+=bert_tocs[search_idx]
                        elif search_str+' '+bert_tocs[search_idx] in ans_string:
                            search_str+=' '+bert_tocs[search_idx]
                        else:
                            found = False
                        
                if (search_str==ans_string) and (i<search_idx):
                    
                    positions_for_each.append((i+1, search_idx))
                    position_start.append(i+1)
                    position_end.append(search_idx)
                    
        if len(positions_for_each) > 0 :
            positions_for_all.extend(positions_for_each)
            all_present_kps.append(kp)
        
        
    assert len(positions_for_all) >= len(all_present_kps)
    
    if len(all_present_kps) == 0:
        return [None,None]
    pos_list = [position_start,position_end]
    pos_set = set(positions_for_all)
    return pos_list,pos_set

def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

def remove_stopwords(text):
    text=[word for word in text if word not in stopword]
    return text