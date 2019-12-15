import sys
sys.path.append('D:\github\python')
sys.path.append('/mnt/d/github/python') #linux wsl
from Cek_typo import cek_typo as ct
from Normalisasi_KBBI import normalisasi_kbbi as nkbi
from modulku import praproses as pps
from modulku import StemNstopW as stm
def praposes(a, normalisasi = True):
    #a = teks
    if normalisasi == True:
        a = pps.gantiKarakter(a)
    a = pps.preprocessing(a)
    a = pps.removePunc(a)
    
    if normalisasi == True:
        a = ct.cek_typo(a)
        a = nkbi.norm_kbbi(a)
    a = stm.stemmer_kata(a)
    a = stm.stop_word(a)
    return a
def savek ():
    nkbi.save_gdiganti()
    ct.save_gdiganti()
    stm.save_kta()
def saves():
    stm.save_kta()
