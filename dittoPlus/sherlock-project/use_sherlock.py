import numpy as np
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

prepare_feature_extraction()
initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()

data = pd.Series(
    [
        ["Jane Smith", "Lute Ahorn", "Anna James"],
        ["Amsterdam", "Haarlem", "Zwolle"],
        ["Chabot Street 19", "1200 fifth Avenue", "Binnenkant 22, 1011BH"]
    ],
    name="values"
)

print(data)

extract_features(
    "../temporary.csv",
    data
)
feature_vectors = pd.read_csv("../temporary.csv", dtype=np.float32)
print(feature_vectors)

model = SherlockModel();
model.initialize_model_from_json(with_weights=True, model_id="sherlock")

predicted_labels = model.predict(feature_vectors, "sherlock")
print(predicted_labels)