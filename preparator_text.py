from nltk.stem import PorterStemmer
import unicodedata
from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize

def PreparatorStem(input):
    ps = PorterStemmer()
    output = []
    for w in input:
        output.append(ps.stem(w))
    return output

def PreparatorTransliterate(input:str):
    """
    ref: https://www.programcreek.com/python/?CodeExample=remove+accents
    """
    text = str(input)
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join([c for c in normalized if not unicodedata.combining(c)])
    return normalized

def PreparatorSyllabify(input:str):
    SSP = SyllableTokenizer()
    return [SSP.tokenize(token) for token in word_tokenize(input)]

def PreparatorAcronymize(input:str):
    acron = ""
    for w in input.split():
        acron += w[0]
    return acron.upper()


def main():
    input = "split the attribute given predefined ranges of the included attributes"
    print(PreparatorStem(input))
    print(PreparatorTransliterate("München"))
    print(PreparatorSyllabify(input))
    print(PreparatorAcronymize("Very Large Data Base"))

if __name__ == '__main__':
    main()
