import re
from preparator_text import PreparatorTransliterate

from sklearn.utils import column_or_1d

# TODO: remove special characters
from pyphonetics import Metaphone
import pandas as pd
import tokenize
# import nltk
import numpy as np



def PreparatorRemoveSpecialCharacters(unicode_line):
    translation_table = dict.fromkeys(map(ord, ",.:;\"^'/\\\\!@#$%&*()_+=|<>?{}\\[]~-"), None)
    unicode_line = unicode_line.translate(translation_table)
    return unicode_line


# TODO: split attributes
def PreparatorSplitAttribute(unicode_line=''):
    fields_border = [0, 15, 28, 30, 43, None]
    fields_name = ['last_name', 'first_name', 'middle_name',
                   'zip_code', 'street_address']
    fields = {}
    # padding input string to length 64
    unicode_line = unicode_line.ljust(64, ' ')
    for i in range(len(fields_border) - 1):
        sub_s = unicode_line[fields_border[i]: fields_border[i + 1]]
        clean_sub_s = cleanStringTokens(sub_s)
        fields[fields_name[i]] = clean_sub_s
    print(fields)
    return fields


def cleanStringTokens(v):
    v = v.replace("\r\n", "\n")
    v = v.replace("\r", "\n")
    v = v.replace("\n", "")
    v = v.replace("\"", "")
    v = v.strip().replace(" +", " ").strip()
    return v


# TODO: merge attributes
def PreparatorMergeAttributes(unicode_line: dict):
    return ' '.join(unicode_line[k] for k in sorted(unicode_line))


def Preparator_MergeAttributes(unicode_line, sep=","):
    return f"{sep}".join(unicode_line)


def PreparatorPhoneticEncode(unicode_line):
    ''' phonetic encode'''
    metaphone = Metaphone()
    word = metaphone.phonetics(unicode_line)
    return word


def PreparatorCapitalize(unicode_line):
    return unicode_line.upper()
 

def main():
    # value = '<>hello world \\<[]'
    # value = PreparatorRemoveSpecialCharacters(value)
    # print(value)
    # uni_input = 'Lan Li 60646'
    # PreparatorSplitAttribute(uni_input)
    # uni_input = {'city': 'Berlin',
    #              'zip': '61820',
    #              'street': 'Nowhere'}
    # res = PreparatorMergeAttributes(uni_input)
    # print(res)
    uni_code = 'Berlin'
    # res = PreparatorPhoneticEncode(uni_code)
    # res = PreparatorCapitalize(uni_code)
    # print(res)
    d = {
        "description":['thiscityis Beijing Ludäscher', 'Isawhimon the street','she was a girl',''], 
        "year":[1987,1099.0,'',1400.0],
        "authors":['Lan Li, Ludäscher Li', 'May, Jane','', 'M Jiang, Li Liang']
    }
    df = pd.DataFrame(d)
    print(type(df))
    df = df.applymap(PreparatorTransliterate)
    print(type(df))
    df['description'] = df['description'].apply(PreparatorRemoveSpecialCharacters)
    print(df['description'])
    df['year'] = df['year'].replace('', 0).astype(float).astype(int) # fillna 
    print(df['year'])

    df['authors'] = df['authors'].apply(lambda x: x.split(','))
    df['authors'] = df['authors'].apply(PreparatorTransliterate)
    df['authors'] = df['authors'].apply(eval)
    print(df['authors'])
    print(type(df['authors'][0]))
    
    df['authors'] = df['authors'].apply(lambda x: Preparator_MergeAttributes(x, sep=','))
    print(df['authors'])


if __name__ == '__main__':
    main()
