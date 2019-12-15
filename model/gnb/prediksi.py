from joblib import load
from Modul import Praproses_data as pps
import pandas as pd
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import json

# import sys
# sys.path.append('D:\github\python')

# load the model from disk TF-idf
filename = dir_path+'/tfdf_model_18.joblib'
tfidf = load(filename)

# load the model from disk SVM
filename = dir_path+'/GNB_MODEL_18.joblib'
GNB = load(filename)

# load the model from disk TF-idf
# filename = dir_path+'/tfdf_model_2.joblib'
# tfidf2 = load(filename)

# # load the model from disk SVM
# filename = dir_path+'/SVM_MODEL_2.joblib'
# SVM2 = load(filename)
with open(dir_path+'/bobot_gnb3.json','r') as f:
    weight = json.load(f)

def prediksi_single(x, normalisasi = True):
    return GNB.predict(tfidf.transform([pps.praposes(x)]).toarray())[0]

def prediksi_list(x, normalisasi = True):
    new_x = list()
    #print("Praproses ->")
    for ix, i in enumerate(x):
        new_x.append(pps.praposes(i, normalisasi = normalisasi))
        
        if ix%100==0 and ix != 0:
            print(ix, end=" ")
        else:
            print(".", end="")
    print("|")
    print("<<Masuk Proses Prediksi>>")
    hasil_prediksi_list = GNB.predict(tfidf.transform(new_x).toarray())
    dixt = {
        "prediksi":list(hasil_prediksi_list),
        "komentar":new_x
    }
    return pd.DataFrame.from_dict(dixt)
def save_kata ():
    pps.savek()

def prediksi(x, normalisasi = True):
    if type(x) is str:
        return GNB.predict(tfidf.transform([pps.praposes(x, normalisasi = normalisasi)]).toarray())[0]
    elif type(x) is list:
        new_x = list()
        #print("Praproses ->")
        for ix, i in enumerate(x):
            new_x.append(pps.praposes(i))
            
            if ix%100==0 and ix != 0:
                print(ix, end=" ")
            else:
                print(".", end="")
        print("|")
        print("<<Masuk Proses Prediksi>>")
        hasil_prediksi_list = GNB.predict(tfidf.transform(new_x).toarray())
        dixt = {
            "prediksi":list(hasil_prediksi_list),
            "komentar":new_x
        }
        return pd.DataFrame.from_dict(dixt)
    else:
        return "Masukan harus bertipe string atau list (array)" 
    
# def prediksi_SVM_list_nopps(x):
#     # new_x = list()
#     # #print("Praproses ->")
#     # for ix, i in enumerate(x):
#     #     new_x.append(pps.praposes(i))
        
#     #     if ix%100==0 and ix != 0:
#     #         print(ix, end=" ")
#     #     else:
#     #         print(".", end="")
#     print("|")
#     print("<<Masuk Proses Prediksi>>")
#     hasil_prediksi_list = SVM.predict(tfidf.transform(x).toarray())
#     dixt = {
#         "prediksi":list(hasil_prediksi_list),
#         "komentar":x
#     }
    # return pd.DataFrame.from_dict(dixt)



# def prediksi_cnb_single(x):
#     return cnb.predict(tfidf_cnb.transform([pps.praposes(x)]).toarray())[0]

# def prediksi_cnb_list(x):
#     new_x = list()
#     #print("Praproses ->")
#     for ix, i in enumerate(x):
#         new_x.append(pps.praposes(i))
        
#         if ix%100==0 and ix != 0:
#             print(ix, end=" ")
#         else:
#             print(".", end="")
#     print("|")
#     print("<<Masuk Proses Prediksi>>")
#     hasil_prediksi_list = cnb.predict(tfidf.transform(new_x).toarray())
#     dixt = {
#         "prediksi":list(hasil_prediksi_list),
#         "komentar":new_x
#     }
#     return pd.DataFrame.from_dict(dixt)

# def prediksi2_single(x):
#     return SVM2.predict(tfidf2.transform([pps.praposes(x)]).toarray())[0]

# def prediksi2_list(x):
#     new_x = list()
#     #print("Praproses ->")
#     for ix, i in enumerate(x):
#         new_x.append(pps.praposes(i))
        
#         if ix%100==0 and ix != 0:
#             print(ix, end=" ")
#         else:
#             print(".", end="")
#     print("|")
#     print("<<Masuk Proses Prediksi>>")
#     hasil_prediksi_list = SVM2.predict(tfidf2.transform(new_x).toarray())
#     dixt = {
#         "prediksi":list(hasil_prediksi_list),
#         "komentar":new_x
#     }
#     return pd.DataFrame.from_dict(dixt)