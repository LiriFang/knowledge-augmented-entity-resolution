import numpy as np
import csv
import sys
import os
import spacy
import re
from collections import Counter
import pandas as pd
import pyarrow as pa

from sherlock import helpers
from sherlock.deploy.model import SherlockModel
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings


class DKInjector:
    """Inject domain knowledge to the data entry pairs.

    Attributes:
        config: the task configuration
        name: the injector name
    """

    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.initialize()

    def initialize(self):
        pass

    def transform(self, entry):
        return entry

    def transform_file(self, input_fn, overwrite=False):
        """Transform all lines of a tsv file.

        Run the knowledge injector. If the output already exists, just return the file name.

        Args:
            input_fn (str): the input file name
            overwrite (bool, optional): if true, then overwrite any cached output

        Returns:
            str: the output file name
        """
        out_fn = input_fn + '.dk'
        if not os.path.exists(out_fn) or \
                os.stat(out_fn).st_size == 0 or overwrite:

            with open(out_fn, 'w') as fout:
                for line in open(input_fn):
                    LL = line.split('\t')
                    if len(LL) == 3:
                        entry0 = self.transform(LL[0])
                        entry1 = self.transform(LL[1])
                        fout.write(entry0 + '\t' + entry1 + '\t' + LL[2])
        return out_fn


class ProductDKInjector(DKInjector):
    """The domain-knowledge injector for product data.
    """

    def initialize(self):
        """Initialize spacy"""
        self.nlp = spacy.load('en_core_web_lg')

    def transform(self, entry):
        """Transform a data entry.

        Use NER to regconize the product-related named entities and
        mark them in the sequence. Normalize the numbers into the same format.

        Args:
            entry (str): the serialized data entry

        Returns:
            str: the transformed entry
        """
        print(entry)
        res = ''
        doc = self.nlp(entry, disable=['tagger', 'parser'])
        ents = doc.ents
        start_indices = {}
        end_indices = {}

        for ent in ents:
            start, end, label = ent.start, ent.end, ent.label_
            if label in ['NORP', 'GPE', 'LOC', 'PERSON', 'PRODUCT']:
                start_indices[start] = 'PRODUCT'
                end_indices[end] = 'PRODUCT'
            if label in ['DATE', 'QUANTITY', 'TIME', 'PERCENT', 'MONEY']:
                start_indices[start] = 'NUM'
                end_indices[end] = 'NUM'

        for idx, token in enumerate(doc):
            if idx in start_indices:
                res += start_indices[idx] + ' '

            # normalizing the numbers
            if token.like_num:
                try:
                    val = float(token.text)
                    if val == round(val):
                        res += '%d ' % (int(val))
                    else:
                        res += '%.2f ' % (val)
                except:
                    res += token.text + ' '
            elif len(token.text) >= 7 and \
                    any([ch.isdigit() for ch in token.text]):
                res += 'ID ' + token.text + ' '
            else:
                res += token.text + ' '
        print(res)
        return res.strip()


class GeneralDKInjector(DKInjector):
    """The domain-knowledge injector for publication and business data.
    """

    def initialize(self):
        """Initialize spacy"""
        self.nlp = spacy.load('en_core_web_lg')

    def transform(self, entry):
        """Transform a data entry.

        Use NER to regconize the product-related named entities and
        mark them in the sequence. Normalize the numbers into the same format.

        Args:
            entry (str): the serialized data entry

        Returns:
            str: the transformed entry
        """
        res = ''
        print(f'before processing, the data entry: {entry}')
        # COL title VAL web caching for database applications with oracle web cache COL authors VAL jordan parker , jesse anton , zheng zeng , lawrence jacobs , tie zhong , xiang liu COL venue VAL sigmod conference COL year VAL 2002
        doc = self.nlp(entry, disable=['tagger', 'parser'])
        print(f'this is the doc after using nlp() with data entry: {doc}')
        # this is the doc after using nlp() with data entry:
        # COL title VAL web caching for database applications with oracle web cache COL authors VAL jordan parker , jesse anton , zheng zeng , lawrence jacobs , tie zhong , xiang liu COL venue VAL sigmod conference COL year VAL 2002
        ents = doc.ents
        print(f'all the entites: {ents}')
        start_indices = {}
        end_indices = {}

        for ent in ents:
            start, end, label = ent.start, ent.end, ent.label_
            print(f'the start: {start}; the end: {end}; the label : {label}')
            if label in ['PERSON', 'ORG', 'LOC', 'PRODUCT', 'DATE', 'QUANTITY', 'TIME']:
                start_indices[start] = label
                end_indices[end] = label

        for idx, token in enumerate(doc):
            if idx in start_indices:
                res += start_indices[idx] + ' '

            # normalizing the numbers
            if token.like_num:
                try:
                    val = float(token.text)
                    if val == round(val):
                        res += '%d ' % (int(val))
                    else:
                        res += '%.2f ' % (val)
                except:
                    res += token.text + ' '
            elif len(token.text) >= 7 and \
                    any([ch.isdigit() for ch in token.text]):
                res += 'ID ' + token.text + ' '
            else:
                res += token.text + ' '
        print(res)
        return res.strip()


class SherlockDKInjector(DKInjector):
    """
    The domain-knowledge inferred by Sherlock
    Deep Learning system

    """

    def initialize(self):
        """Initialize spacy"""
        # self.nlp = spacy.load('en_core_web_lg')
        prepare_feature_extraction()
        initialise_word_embeddings()
        initialise_pretrained_model(400)
        # 400 => dimension
        initialise_nltk()

        self.model = SherlockModel()
        self.model.initialize_model_from_json(with_weights=True, model_id="sherlock")

    def sep_ds(self, col_names, ds):
        """
        Separate the combined and serialized dataset to the original datasets
        use this as the input for Sherlock
        """
        dict_prep = dict.fromkeys(col_names, [])
        for list_v in ds:
            for col_name_p, col_value_p in list_v:
                # print({col_name.strip(): col_value})
                # print(col_value)
                col_name = col_name_p.strip()
                col_value = col_value_p.strip()
                dict_prep[col_name].append(col_value)
        return pd.DataFrame(dict_prep)

    def create_input_ds(self, ds_f):
        col_names_1 = []
        col_names_2 = []
        ds_1 = []
        ds_2 = []
        ds_3 = []

        for line_p in open(ds_f):
            pattern = re.compile(r'(COL|VAL)')
            line = line_p.rstrip()
            body = line[:-1]
            tail = line[-1]
            print(tail)
            items = pattern.split(body)[1:]
            assert len(items) % 4 == 0
            pairs = []
            for i in range(len(items) // 4):
                pairs.append((items[i * 4 + 1], items[i * 4 + 3]))
            pair_1 = pairs[:len(pairs) // 2]
            pair_2 = pairs[len(pairs) // 2:]
            col_names_1 = [v[0].strip() for v in pair_1]
            col_names_2 = [v[0].strip() for v in pair_2]
            ds_1.append(pair_1)
            ds_2.append(pair_2)
        df_1 = self.sep_ds(col_names_1, ds_1)
        df_2 = self.sep_ds(col_names_2, ds_2)
        df_3 = pd.DataFrame({'flag': ds_3})
        return df_1, df_2, df_3

    def prev_transform(self, df, cols, new_df):
        """
        Before combining two datasets:
        Manually serialized the rows
        Save as a new DataFramew
        """
        for index, row in df.iterrows():
            for col in cols:
                old_value = row[col]
                # TODO: There is no need to do any "normalization" in this step
                # WE have already prepared the dataset
                new_value = f"COL {col} NUM VAL {old_value}"
                print(new_value)
                new_df.at[index, col] = new_value
                # .....new_df
        return new_df

    def transform_file(self, input_fn, overwrite=False):
        """Transform all lines of a tsv file.

        Run the knowledge injector. If the output already exists, just return the file name.

        Args:
            input_fn (str): the input file name
            overwrite (bool, optional): if true, then overwrite any cached output

        Returns:
            str: the output file name
        """
        out_fn = input_fn + '.dk'
        if not os.path.exists(out_fn) or \
                os.stat(out_fn).st_size == 0 or overwrite:

            with open(out_fn, 'w') as fout:
                df1, df2, df3 = self.create_input_ds(input_fn)  # the first dataset, the second dataset, and the flag
                # Use Sherlock to predict the column types
                # then annotate each cell across the columns:
                # returns: list of predicted labels: e.g., array(['person', 'city', 'address'], dtype=object)
                # df1: the first dataset; df2: the second dataset; df3: the pairing result

                extract_features(
                    "../temporary_1.csv",
                    df1
                )
                feature_vectors_1 = pd.read_csv("../temporary_1.csv", dtype=np.float32)
                # print(feature_vectors_1)
                predicted_labels_1 = model.predict(feature_vectors_1, "sherlock")
                print(predicted_labels_1)
                raise NotImplementedError

                extract_features(
                    "../temporary_2.csv",
                    df2
                )
                feature_vectors_2 = pd.read_csv("../temporary_2.csv", dtype=np.float32)
                predicted_labels_2 = model.predict(feature_vectors_2, "sherlock")
                print(predicted_labels_2)

                cols_1 = list(df1.columns)
                annotate_df1 = pd.DataFrame(columns=cols_1)  # embed first
                df1_serialized = self.prev_transform(df1, cols_1, annotate_df1)

                cols_2 = list(df2.columns)
                annotate_df2 = pd.DataFrame(columns=cols_2)  # embed first
                df2_serialized = self.prev_transform(df2, cols_2, annotate_df2)
                assert len(df1_serialized) == len(df2_serialized)
                for i in range(len(new_dfl)):
                    entry0 = ''
                    entry1 = ''
                    fir_row = df1_serialized.iloc[i]
                    entry0 += ' '.join(fir_row)
                    sec_row = df2_serialized.iloc[i]
                    entry1 += ' '.join(sec_row)
                    fout.write(entry0 + '\t' + entry1 + '\t' + LL[2])
        return out_fn

