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