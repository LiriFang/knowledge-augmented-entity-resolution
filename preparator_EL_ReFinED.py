from refined.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model', 
                                  entity_set="wikipedia",
                                  data_dir="data/refined/wikipedia/", 
                                  download_files=True,
                                  use_precomputed_descriptions=True,
                                  device="cuda:0")

# ents = refined.process_text("England won the FIFA World Cup in 1966.")
# print([(ent.text, ent.pred_entity_id, ent.pred_types) for ent in ents]) 
# [('England', ({'wikidata_qcode': 'Q47762'}, 0.7631), [('Q9332', 'behavior', 0.998), 
# ('Q104637332', 'planned process', 0.999), ('Q12973014', 'sports team', 0.9987)]), 
# ('FIFA World Cup', ({'wikidata_qcode': 'Q19317'}, 0.9505), [('Q4026292', 'action', 0.8019), 
# ('Q500834', 'tournament', 0.9996)])]


