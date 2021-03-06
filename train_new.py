import pandas as pd
import hashlib, base64
import datetime

from StemCharLstmCrfModel import StemCharLstmCrfModel
from LemmaCharLstmCrfModel import LemmaCharLstmCrfModel
from ElmoLstmCrfModel import ElmoLstmCrfModel

BATCH_SIZE = 32
EPOCHS = 13
MAX_LEN = 75
MAX_LEN_CHARS = 15
EMBEDDING = 90
EMBEDDING_WORD = 30


#data = pd.read_csv("/Users/kpentchev/data/teo_tagged_3_fixed.csv", encoding="utf-8", delimiter='\t')
data = pd.read_csv("/Users/kpentchev/data/ner_2019_04_06_fixed.csv", encoding="utf-8", delimiter='\t', quoting=3)
#data = pd.read_csv("/home/kpentchev/data/floyd/ner_2019_03_11_no_med.csv", encoding="utf-8", delimiter='\t')
#data = pd.read_csv("/home/kpentchev/data/floyd/teo_tagged_2019_02_11.csv", encoding="utf-8", delimiter='\t')
data = data.fillna(method="ffill")

model = StemCharLstmCrfModel(EMBEDDING, EMBEDDING_WORD, MAX_LEN, MAX_LEN_CHARS)
#model = LemmaCharLstmCrfModel(EMBEDDING, EMBEDDING_WORD, MAX_LEN, MAX_LEN_CHARS)
#model = ElmoLstmCrfModel(MAX_LEN)

model.train(data, BATCH_SIZE, EPOCHS, 0.1)

model.evaluate()
    
# Saving Model
model.save('/Users/kpentchev/data/models/{}_stem_char_lstm_crf.h5'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')))
