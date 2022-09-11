import argparse
import pandas as pd
from doduo.doduo import Doduo

# Load Doduo model
args = argparse.Namespace
args.model = "wikitable"  # or args.model = "viznet"
doduo = Doduo(args)

# Load sample tables
df1 = pd.read_csv("../data-preparator-for-EM/data/Abt-Buy/test-1.csv", index_col=0)
df2 = pd.read_csv("../data-preparator-for-EM/data/Abt-Buy/test-2.csv", index_col=0)

# Sample 1: Column annotation
annot_df1 = doduo.annotate_columns(df1)
print(annot_df1.coltypes)
print(annot_df1.colrels)


# Sample 2: Column annotation
annot_df2 = doduo.annotate_columns(df2)
print(annot_df2.coltypes)
print(annot_df2.colrels)
