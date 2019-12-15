import json
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(dir_path+"/dict_fitur_non_spam.json",'r') as f:
    dict_fitur_non_spam = json.load(f)
with open(dir_path+"/dict_fitur_spam.json",'r') as f:
    dict_fitur_spam = json.load(f)

with open(dir_path+"/frequensi.json",'r') as f:
    dict_fitur = json.load(f)

def frequensi_ (kata):
    if kata in dict_fitur_non_spam:
        non_spam = dict_fitur_non_spam[kata]
    else:
        non_spam = 0
    if kata in dict_fitur_spam:
        spam = dict_fitur_spam[kata]
    else:
        spam = 0
    return {kata:{"spam":spam, "non_spam":non_spam}}

def frequensi (kata):
    if kata in dict_fitur:
        ket = {kata:dict_fitur[kata]}
    else:
        ket = {kata:{"spam":0, "non_spam":0}}
    return ket