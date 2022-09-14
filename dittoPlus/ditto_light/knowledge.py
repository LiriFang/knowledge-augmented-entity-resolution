import numpy as np
import csv
import sys
import os
import re

import pandas as pd
import pyarrow as pa 

import spacy

from collections import Counter

from tqdm import tqdm

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
                for line in tqdm(open(input_fn)):
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
        doc = self.nlp(entry, disable=['tagger', 'parser'])
        ents = doc.ents
        start_indices = {}
        end_indices = {}

        for ent in ents:
            start, end, label = ent.start, ent.end, ent.label_
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
        return res.strip()


class EntityLinkingDKInjector(DKInjector):
    """The domain-knowledge injector for publication and business data.
    """
    def initialize(self):
        """Initialize EL model"""
        # self.nlp = spacy.load('en_core_web_lg')

        # load refined model here
        from refined.processor import Refined
        print("Loading RefinED model...")
        self.refined = Refined.from_pretrained(model_name='wikipedia_model', 
                            entity_set="wikipedia",
                            data_dir="/home/yirenl2/PLM_DC/data-preparator-for-EM/data/refined/wikipedia", 
                            download_files=True,
                            use_precomputed_descriptions=True,
                            device="cuda:0",
        )

        self.log_path = 'output/refined_outputs.txt'
        self.log_file = open(self.log_path, 'w')

    def transform(self, entry):
        """Transform a data entry.

        Use NER to regconize the product-related named entities and
        mark them in the sequence. Normalize the numbers into the same format.

        Args:
            entry (str): the serialized data entry

        Returns:
            str: the transformed entry
        """

        # print(entry)
        # COL name VAL lg 24 ' lds4821ww semi integrated built in white dishwasher 
        # lds4821wh 
        # COL description VAL lg 24 ' lds4821ww semi integrated built in 
        # white dishwasher lds4821wh xl tall tub cleans up to 16 place settings at once 
        # adjustable upper rack lodecibel quiet operation senseclean wash system 4 wash 
        # cycles with 3 spray arms multi-level water direction slim direct drive motor 
        # semi-integrated electronic control panel white finish 
        # COL price VAL  
        # print([t.split('VAL ')[-1] for t in entry.split('COL ')][1:])

        cols = [t.split(' VAL')[0] for t in entry.split('COL ')][1:]
        values = [t.split('VAL ')[-1] for t in entry.split('COL ')][1:]
        # print(cols)

        valuesTagged = []
        for entry in values:
            if len(entry.replace(' ', '')) > 0:
                # print(entry)
                # print(len(entry))
                spans = self.refined.process_text(entry)
                text = entry
                for i in range(len(spans)-1, -1, -1):
                    span = spans[i]
                    if len(span.pred_types) > 0:
                        spanType = span.pred_types[0][1]
                        # text = text[:span.start] + '<' + spanType + '>' + entry[span.start:span.start + span.ln] + '</' + spanType + '>' + entry[span.start + span.ln:]
                        text = text[:span.start] + entry[span.start:span.start + span.ln] + ' (' + spanType + ')' + entry[span.start + span.ln:]
                    else:
                        text = entry
                valuesTagged.append(text)
            else:
                valuesTagged.append("")
        
        res = ""
        for idx, col in enumerate(cols):
            res += "COL %s VAL %s "%(col, valuesTagged[idx])
        # print(res)
        self.log_file.write(res + '\n')
        return res.strip()
        # raise NotImplementedError
        


class SherlockDKInjector(DKInjector):
    """
    The domain-knowledge inferred by Sherlock
    Deep Learning system 

    """
    def initialize(self):
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

        """Initialize spacy"""
        # self.nlp = spacy.load('en_core_web_lg')
        prepare_feature_extraction()
        initialise_word_embeddings()
        initialise_pretrained_model(400)
        # 400 => dimension 
        initialise_nltk()

    def sep_ds(self, col_names, ds):
        """
        Separate the combined and serialized dataset to the original datasets
        use this as the input for Sherlock
        """
        dict_prep = {}
        for inner_value in ds:
            # [(' title ', ' query optimization by predicate move-around '), (' authors ', ' inderpal singh mumick , alon y. levy , yehoshua sagiv '), (' venue ', ' vldb '), (' year ', ' 1994 \t')]
            for k, v in inner_value:
                col_name = k.strip()
                col_value = v.strip()
                dict_prep.setdefault(col_name, []).append(col_value)
        return pd.DataFrame(dict_prep)

    def create_input_ds(self, ds_f):
        col_names_1 = []
        col_names_2 = []
        ds_1 = []
        ds_2 = []
        ds_3 = []
        tails = []
        for line_p in open(ds_f):
            pattern = re.compile(r'(COL|VAL)')
            line = line_p.rstrip()
            body = line[:-1]
            tail = line[-1]
            tails.append(int(tail))
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
        df_3 = pd.DataFrame({'flag': tails})
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
                # print(new_value)
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
                df1, df2, df3 = self.create_input_ds(input_fn) # the first dataset, the second dataset, and the flag
                # Use Sherlock to predict the column types 
                # then annotate each cell across the columns: 
                # returns: list of predicted labels: e.g., array(['person', 'city', 'address'], dtype=object)
                # df1: the first dataset; df2: the second dataset; df3: the pairing result
                # for index, row in df1.iterrows():
                #     print(row)

                model_1 = SherlockModel()
                model_1.initialize_model_from_json(with_weights=True, model_id="sherlock")

                extract_features(
                    "../temporary_1.csv",
                    df1
                )
                feature_vectors_1 = pd.read_csv("../temporary_1.csv", dtype=np.float32)
                print(feature_vectors_1)
                print(len(feature_vectors_1))

                raise NotImplementedError
                predicted_labels_1 = model_1.predict(feature_vectors_1, "sherlock")
                
                extract_features(
                    "../temporary_2.csv",
                    df2
                )
                model_2 = SherlockModel()
                model_2.initialize_model_from_json(with_weights=True, model_id="sherlock")

                feature_vectors_2 = pd.read_csv("../temporary_2.csv", dtype=np.float32)
                predicted_labels_2 = model_2.predict(feature_vectors_2, "sherlock")

                cols_1 = list(df1.columns)
                annotate_df1 = pd.DataFrame(columns=cols_1) # embed first 
                df1_serialized = self.prev_transform(df1, cols_1, annotate_df1)
                print(cols_1)
                print(predicted_labels_1)
                raise NotImplementedError
                # print(df1.ABV)

                cols_2 = list(df2.columns)
                annotate_df2 = pd.DataFrame(columns=cols_2) # embed first 
                df2_serialized = self.prev_transform(df2, cols_2, annotate_df2)

                # print(cols_2)
                # print(predicted_labels_2)
                # raise NotImplementedError

                assert len(df1_serialized) == len(df2_serialized)
                for i in range(len(df1_serialized)):
                    entry0 = ''
                    entry1 = ''
                    fir_row = df1_serialized.iloc[i]
                    entry0 += ' '.join(fir_row)
                    sec_row = df2_serialized.iloc[i]
                    entry1 += ' '.join(sec_row)
                    entry2 = int(df3.loc[i, 'flag'])
                    fout.write(entry0 + '\t' + entry1 + '\t' + str(entry2) + '\n')
                    # print(f"{entry0} + '\t' + {entry1} + '\t' + {entry2}")
        return out_fn

