from refined.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model', 
                                  entity_set="wikipedia",
                                  data_dir="/path/to/download/data/to/", 
                                  download_files=True,
                                  use_precomputed_descriptions=True,
                                  device="cuda:0")

ents = refined.process_text("England won the FIFA World Cup in 1966.")

print([(ent.text, ent.pred_entity_id, ent.pred_types) for ent in ents])