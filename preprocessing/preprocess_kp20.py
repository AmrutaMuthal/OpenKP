import tensorflow as tf
import numpy as np
from tokenizers import BertWordPieceTokenizer
import os
import sys
sys.path.append('../OpenKP/') 
from utils import extractionUtils as ext
import pandas as pd

import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

import pickle

sentence_re = r'''(?x)      # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''

vocab = "D:/Word embedding/bert/assets/vocab.txt"
tokenizer = BertWordPieceTokenizer(vocab, lowercase=True)
kp_training = pd.read_json('kp/kp20k_training.json', lines=True)
kp_testing = pd.read_json('kp/kp20k_testing.json', lines=True)
kp_validation = pd.read_json('kp/kp20k_validation.json', lines=True)

max_kp = 0
min_len =1e100
max_len = 512

all_reps = []
att_masks = []
key_positions = []
ref_positions = []

for i in range(len(kp_training[:20000])):
    text = kp_training['abstract'][i]
    txt = ext.remove_punctuation(text)
    text_toc = nltk.regexp_tokenize(txt, sentence_re)
    bert_toc_idx = tokenizer.encode(txt)
    bert_tocs = [txt[start:end] for start,end in bert_toc_idx.offsets]
    padding_length = max_len - len(bert_tocs)
    rep = bert_toc_idx.ids


    if padding_length<0:
        rep = np.array(rep[:max_len])
        attention_mask = np.array([1]*max_len)
    else:
        attention_mask = np.array(([1]*len(rep))+([0]*padding_length))
        rep = np.array(rep + ([0]*padding_length))


    kps = ext.get_kp(text_toc)
    pos,pos_set = ext.find_positions(text_toc,bert_tocs,kps)
    
    if pos_set:
        can = kp_training['keyword'][i].split(';')
        can = [line.split() for line in can]
        if len(can[0])<1:
            print(len(can[0]))
            continue
        ref_pos,ref_set = ext.find_positions(text_toc,bert_tocs,can)

    if pos_set and ref_set:
        key_positions.append(pos_set)
        #rep = np.zeros((len(text_toc),300),dtype=float)
        #rep[idx] = model2[np.array(text_toc)[idx]]  
        all_reps.append(rep)
        att_masks.append(attention_mask)
        #max_len = max(max_len,len(text_toc))
        #min_len = min(min_len,len(text_toc))
        max_kp = max(max_kp,len(pos_set))
        ref_positions.append(ref_set)
        
y_label = []
final_positions = []
pos_mask = []
final_kp_list = []
op_mask = []
for idx,kp in enumerate(key_positions):
    kp = list(kp)
    start = []
    end = []
    y_val = []
    #y_val = [2 if (key in ref_positions[idx] and key[1]<512) else 1 for key in kp]
    for key in kp:
        if key[1]<512:
            start.append(key[0]-1)
            end.append(key[1]-1)
            y_val.append([2,1] if key in ref_positions[idx] else [1,2])
            
    y_val.extend([[0,0]]*(max_kp-len(start)))
    op_mask.append([[True,True]]*len(start)+[[False,False]]*(max_kp-len(start)))
    final_positions.append(tf.pad([start,end],[[0,0],[0,max_kp-len(start)]]))
    pos_mask.append([True]*len(start)+[False]*(max_kp-len(start)))
    
    y_label.append(y_val)
    final_kp_list.append(kp)
    #print(idx,len(y_val),max_kp-len(start))
    
x_train = tf.transpose(all_reps,perm=[0,1])
x_mask = tf.transpose(att_masks,perm=[0,1])
x_pos = tf.stack(final_positions)
y_train = tf.stack(y_label)
y_mask = tf.stack(op_mask)
y_train = tf.cast(y_train,dtype=float)

np.savetxt('Data/KP20/x_train.csv',x_train)
np.savetxt('Data/KP20/x_mask_train.csv',x_mask)
with open('Data/KP20/x_pos_train.pkl','wb') as f:
    pickle.dump(x_pos, f)

with open('Data/KP20/y_mask_train.pkl','wb') as f:
    pickle.dump(y_mask, f)
    
with open('Data/KP20/y_train.pkl','wb') as f:
    pickle.dump(y_train, f)

del x_train
del x_mask
del x_pos
del y_train
del y_mask
            
############################## Testing data processing###################################

all_reps = []
att_masks = []
key_positions = []
ref_positions = []

for i in range(len(kp_testing)):
    text = kp_testing['abstract'][i]
    txt = ext.remove_punctuation(text)
    text_toc = nltk.regexp_tokenize(txt, sentence_re)
    bert_toc_idx = tokenizer.encode(txt)
    bert_tocs = [txt[start:end] for start,end in bert_toc_idx.offsets]
    padding_length = max_len - len(bert_tocs)
    rep = bert_toc_idx.ids


    if padding_length<0:
        rep = np.array(rep[:max_len])
        attention_mask = np.array([1]*max_len)
    else:
        attention_mask = np.array(([1]*len(rep))+([0]*padding_length))
        rep = np.array(rep + ([0]*padding_length))


    kps = ext.get_kp(text_toc)
    pos,pos_set = ext.find_positions(text_toc,bert_tocs,kps)
    if len(kps)==0:
        continue
    if pos_set:
        can = kp_testing['keyword'][i].split(';')
        can = [line.split() for line in can]
        if len(can[0])<1:
            continue
        ref_pos,ref_set = ext.find_positions(text_toc,bert_tocs,can)

    if pos_set and ref_set:
        key_positions.append(pos_set)
        #rep = np.zeros((len(text_toc),300),dtype=float)
        #rep[idx] = model2[np.array(text_toc)[idx]]  
        all_reps.append(rep)
        att_masks.append(attention_mask)
        #max_len = max(max_len,len(text_toc))
        #min_len = min(min_len,len(text_toc))
        #max_kp = max(max_kp,len(pos_set))
        ref_positions.append(ref_set)
        
y_label = []
final_positions = []
pos_mask = []
final_kp_list = []
op_mask = []
for idx,kp in enumerate(key_positions):
    kp = list(kp)
    start = []
    end = []
    y_val = []
    #y_val = [2 if (key in ref_positions[idx] and key[1]<512) else 1 for key in kp]
    
    for num, key in enumerate(kp):
        if (key[1]<512) and num<max_kp:
            start.append(key[0]-1)
            end.append(key[1]-1)
            y_val.append([2,1] if key in ref_positions[idx] else [1,2])
            
    y_val.extend([[0,0]]*(max_kp-len(start)))
    op_mask.append([[True,True]]*len(start)+[[False,False]]*(max_kp-len(start)))
    final_positions.append(tf.pad([start,end],[[0,0],[0,max_kp-len(start)]]))
    pos_mask.append([True]*len(start)+[False]*(max_kp-len(start)))
    
    y_label.append(y_val)
    final_kp_list.append(kp)
    #print(idx,len(y_val),max_kp-len(start))
    
x_test = tf.transpose(all_reps,perm=[0,1])
x_mask = tf.transpose(att_masks,perm=[0,1])
x_pos = tf.stack(final_positions)
y_test = tf.stack(y_label)
y_mask = tf.stack(op_mask)
y_test = tf.cast(y_test,dtype=float)

np.savetxt('Data/KP20/x_test.csv',x_test)
np.savetxt('Data/KP20/x_mask_test.csv',x_mask)
with open('Data/KP20/x_pos_test.pkl','wb') as f:
    pickle.dump(x_pos, f)

with open('Data/KP20/y_mask_test.pkl','wb') as f:
    pickle.dump(y_mask, f)
    
with open('Data/KP20/y_test.pkl','wb') as f:
    pickle.dump(y_test, f)
    
del x_test
del x_mask
del x_pos
del y_test
del y_mask

######################################## Validation data processing########################################

all_reps = []
att_masks = []
key_positions = []
ref_positions = []

for i in range(len(kp_validation)):
    text = kp_validation['abstract'][i]
    txt = ext.remove_punctuation(text)
    text_toc = nltk.regexp_tokenize(txt, sentence_re)
    bert_toc_idx = tokenizer.encode(txt)
    bert_tocs = [txt[start:end] for start,end in bert_toc_idx.offsets]
    padding_length = max_len - len(bert_tocs)
    rep = bert_toc_idx.ids


    if padding_length<0:
        rep = np.array(rep[:max_len])
        attention_mask = np.array([1]*max_len)
    else:
        attention_mask = np.array(([1]*len(rep))+([0]*padding_length))
        rep = np.array(rep + ([0]*padding_length))


    kps = ext.get_kp(text_toc)
    pos,pos_set = ext.find_positions(text_toc,bert_tocs,kps)
    if len(kps)<1:
        continue
    if pos_set:
        can = kp_validation['keyword'][i].split(';')
        can = [line.split() for line in can]
        if len(can[0])<1:
            continue
        ref_pos,ref_set = ext.find_positions(text_toc,bert_tocs,can)


    if pos_set and ref_set:
        key_positions.append(pos_set)
        #rep = np.zeros((len(text_toc),300),dtype=float)
        #rep[idx] = model2[np.array(text_toc)[idx]]  
        all_reps.append(rep)
        att_masks.append(attention_mask)
        #max_len = max(max_len,len(text_toc))
        #min_len = min(min_len,len(text_toc))
        #max_kp = max(max_kp,len(pos_set))
        ref_positions.append(ref_set)
        
y_label = []
final_positions = []
pos_mask = []
final_kp_list = []
op_mask = []
for idx,kp in enumerate(key_positions):
    kp = list(kp)
    start = []
    end = []
    y_val = []
    #y_val = [2 if (key in ref_positions[idx] and key[1]<512) else 1 for key in kp]
    for num, key in enumerate(kp):
        if (key[1]<512) and num<max_kp:
            start.append(key[0]-1)
            end.append(key[1]-1)
            y_val.append([2,1] if key in ref_positions[idx] else [1,2])
            
    y_val.extend([[0,0]]*(max_kp-len(start)))
    op_mask.append([[True,True]]*len(start)+[[False,False]]*(max_kp-len(start)))
    final_positions.append(tf.pad([start,end],[[0,0],[0,max_kp-len(start)]]))
    pos_mask.append([True]*len(start)+[False]*(max_kp-len(start)))
    
    y_label.append(y_val)
    final_kp_list.append(kp)
    #print(idx,len(y_val),max_kp-len(start))
    
x_val = tf.transpose(all_reps,perm=[0,1])
x_mask = tf.transpose(att_masks,perm=[0,1])
x_pos = tf.stack(final_positions)
y_val = tf.stack(y_label)
y_mask = tf.stack(op_mask)
y_val = tf.cast(y_val,dtype=float)

np.savetxt('Data/KP20/x_val.csv',x_val)
np.savetxt('Data/KP20/x_mask_val.csv',x_mask)
with open('Data/KP20/x_pos_val.pkl','wb') as f:
    pickle.dump(x_pos, f)

with open('Data/KP20/y_mask_val.pkl','wb') as f:
    pickle.dump(y_mask, f)
    
with open('Data/KP20/y_val.pkl','wb') as f:
    pickle.dump(y_val, f)

del x_val
del x_mask
del x_pos
del y_val
del y_mask

print('Max embedding length:', max_len)
print('Max no. of candidate KPs is', max_kp)
