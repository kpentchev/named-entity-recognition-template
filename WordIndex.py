import pickle

class WordIndex(object):
    """Class representing a dictionary"""
    def __init__(self):
        self.pad = "PAD"
        self.unknown = "UNK"
        self.dict = {}
        self.dict[self.unknown] = 1 # Unknown words
        self.dict[self.pad] = 0 # Padding

        self.inverse = {}
        self.inverse[0] = self.pad
        self.inverse[1] = self.unknown

        self.length = 2

    def getIdx(self, word):
        if (not word in self.dict):
            self.length += 1
            self.dict[word] = self.length
            self.inverse[self.length] = word
        return self.dict[word]

    def getWord(self, idx):
        return self.inverse[idx]

    def getPadIdx(self):
        return 0

    def getPad(self):
        return self.pad

    def add(self, words):
        for word in words:
            if (not word in self.dict):
                self.length += 1
                self.dict[word] = self.length
                self.inverse[self.length] = word
                

    def save(self, file):
        with open(file, 'wb') as handle:
            pickle.dump(self.dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file):
        with open(file, 'rb') as handle:
            self.dict = pickle.load(handle)
            self.length = len(self.dict.keys())