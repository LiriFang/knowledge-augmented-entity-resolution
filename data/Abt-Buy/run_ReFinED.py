import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

from refined.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model', 
                            entity_set="wikipedia",
                            data_dir="/home/yirenl2/PLM_DC/data-preparator-for-EM/data/refined/wikipedia", 
                            download_files=True,
                            use_precomputed_descriptions=True,
                            device="cuda:0")


def main():
    splits = ['train', 'valid', 'test']
    for s in splits:
        df = pd.read_csv("%s.txt"%s, sep='\t', header=None)
        df.columns = ['r1', 'r2', 'isDup']
        # print(df.head(2))
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            res = {
                'r1': refined.process_text(row['r1']),
                'r2': refined.process_text(row['r2']),
            }
            results.append(res)
        
        pickle.dump(results, open("%s_refined_outputs.pkl"%s,"wb"))


if __name__ == "__main__":
    main()

    # text = "g 24 ' lds4821ww semi integrated built in white dishwasher lds4821wh xl tall tub cleans up to 16 place settings at once adjustable upper rack lodecibel quiet operation senseclean wash system 4 wash cycles with 3 spray arms multi-level water direction slim direct drive motor semi-integrated electronic control panel white finish"
    # refined = Refined.from_pretrained(model_name='wikipedia_model', 
    #                               entity_set="wikipedia",
    #                               data_dir="/home/yirenl2/PLM_DC/data-preparator-for-EM/data/refined/wikipedia", 
    #                               download_files=True,
    #                               use_precomputed_descriptions=True,
    #                               device="cuda:0")
    # ents = refined.process_text(text)
    # # ents = refined.process_text(text, return_special_spans=True)
    # # print([(ent.text, ent.pred_entity_id, ent.pred_types, ent.coarse_type) for ent in ents]) 
    # # print([ent for ent in ents]) 
    # print([ent for ent in ents][0]) 