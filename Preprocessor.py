from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def encodeSentences(sentences, word2idx):
    # Convert each sentence from list of Token to list of word_index
    encoded = [[word2idx.getIdx(w[0]) for w in s] for s in sentences]
    return encoded

def pad(sentences, maxLength, pad):
    # Padding each sentence to have the same lenght
    padded = pad_sequences(maxlen=maxLength, sequences=sentences, padding="post", value=pad)
    return padded

def encodeTags(tags, tag2idx):
    # Convert Tag/Label to tag_index
    encoded = [[tag2idx.getIdx(w[2]) for w in s] for s in tags]
    return encoded

def onehotEncodeTags(tags, nTags):
    # One-Hot encode
    hotencoded = [to_categorical(i, num_classes=nTags+1) for i in tags]  # n_tags+1(PAD)
    return hotencoded