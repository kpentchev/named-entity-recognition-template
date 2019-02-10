def buildWordIdx(words):
    # Vocabulary Key:word -> Value:token_index
    # The first 2 entries are reserved for PAD and UNK
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1 # Unknown words
    word2idx["PAD"] = 0 # Padding
    return word2idx

def inverseWordIdx(word2idx):
    # Vocabulary Key:token_index -> Value:word
    idx2words = {i: w for w, i in word2idx.items()}
    return idx2words

def buildTagIdx(tags):
    # Vocabulary Key:Label/Tag -> Value:tag_index
    # The first entry is reserved for PAD
    tag2idx = {t: i+1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    return tag2idx

def inverseTagIdx(tag2idx):
    # Vocabulary Key:tag_index -> Value:Label/Tag
    idx2tags = {i: w for w, i in tag2idx.items()}
    return idx2tags