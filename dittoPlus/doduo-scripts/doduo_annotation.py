import argparse
import pandas as pd
from doduo.doduo import Doduo
from collections import defaultdict

def method3(input_fn:str):
    col_names_1 = []
    col_names_2 = []
    ds_1 = defaultdict(list)
    ds_2 = defaultdict(list)
    i = 0

    for line_p in open(input_fn, encoding='utf-8'):
        line = line_p.strip()
        items = line.split('\t')
        # print(len(items))
        col_names_1, val_list_1 = str2col(items[0])
        col_names_2, val_list_2 = str2col(items[1])

        ds_1[i] = val_list_1
        ds_2[i] = val_list_2
        i+=1
    df_1 = pd.DataFrame.from_dict(ds_1, orient='index', columns=col_names_1)
    df_2 = pd.DataFrame.from_dict(ds_2, orient='index', columns=col_names_2)
    # prefix = input_fn.split('.')[0]
    # df_1.to_csv(prefix+'-1.csv')
    # df_2.to_csv(prefix+'-2.csv')
    return df_1, df_2


def str2col(s):
    col_names = []
    val_list = []
    columns = s.split('COL')
    # print(columns)
    for col in columns[1:]:
        vals = col.split('VAL')
        # print(vals)
        col_name = vals[0].strip()
        val = vals[1].strip()


        col_names.append(col_name)
        val_list.append(val)
    assert len(col_names) == len(val_list), "number of col_name is different from the number of vals"
    return col_names, val_list

def col2str(df1, df2, annotate_col_1, annotate_col_2, output_file):
    col_names_1 = list(df1.columns)
    col_names_2 = list(df2.columns)
    with open(output_file, 'w', encoding='utf-8') as f:
        for items_1, items_2 in zip(df1.itertuples(index=False, name=None), df2.itertuples(index=False, name=None)):
            line1 = ' '.join(map(lambda x, y, z: f'COL {x} {y} VAL {z}', col_names_1, annotate_col_1, items_1))
            line2 = ' '.join(map(lambda x, y, z: f'COL {x} {y} VAL {z}', col_names_2, annotate_col_2, items_2))
            f.writelines(line1 + '\t' + line2 + '\n')


file = "C:/Users/fangl/Desktop/ditto-master/data-preparator-for-EM/data/er_magellan/Structured/DBLP-GoogleScholar/valid.txt"
df_1, df_2 = method3(file)
# Load Doduo model
args = argparse.Namespace
args.model = "wikitable"  # or args.model = "viznet"
doduo = Doduo(args)

# Load sample tables
# df1 = pd.read_csv("../data-preparator-for-EM/data/Abt-Buy/test-1.csv", index_col=0)
# df2 = pd.read_csv("../data-preparator-for-EM/data/Abt-Buy/test-2.csv", index_col=0)

# Sample 1: Column annotation
annot_df1 = doduo.annotate_columns(df_1)
print(annot_df1.coltypes)
print(annot_df1.colrels)


# Sample 2: Column annotation
annot_df2 = doduo.annotate_columns(df_2)
print(annot_df2.coltypes)
print(annot_df2.colrels)

col2str(df_1, df_2, annot_df1.coltypes, annot_df2.coltypes, file+'.doduo')
