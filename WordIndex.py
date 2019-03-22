import pickle

class WordIndex(object):
    """Class representing a dictionary"""
    def __init__(self, unknown, maxLength=10000000):
        self.pad = "PAD"
        self.unknown = unknown
        self.dict = {}
        self.dict[self.unknown] = 1 # Unknown words
        self.dict[self.pad] = 0 # Padding

        self.inverse = {}
        self.inverse[0] = self.pad
        self.inverse[1] = self.unknown

        self.length = 2
        self.maxLength = maxLength

    def getIdx(self, word):
        if (not word in self.dict):
            if self.length < self.maxLength -1:
                self.dict[word] = self.length
                self.inverse[self.length] = word
                self.length += 1
            else:
                return self.dict[self.unknown]
        return self.dict[word]

    def getWord(self, idx):
        return self.inverse[idx]

    def getPadIdx(self):
        return 0

    def getPad(self):
        return self.pad

    def getLength(self):
        return self.length

    def add(self, words):
        for word in words:
            if (not word in self.dict):
                self.dict[word] = self.length
                self.inverse[self.length] = word
                self.length += 1

    def save(self, file):
        with open(file, 'wb') as handle:
            pickle.dump(self.dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file):
        with open(file, 'rb') as handle:
            self.dict = pickle.load(handle)
            self.length = len(self.dict.keys())
            self.inverse = {i: w for w, i in self.dict.items()}