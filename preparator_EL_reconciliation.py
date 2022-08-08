import os
import json
from unittest import result

import spacy
import numpy as np

import requests
from tqdm import tqdm

NLP = spacy.load('en_core_web_lg')
RECON_ENDPOINT = 'https://wikidata.reconci.link/en/api'


def get_NER_entities(text):
    doc = NLP(text, disable=['parser'])
    entities = []
    labels = []
    for ent in doc.ents:
        entities.append(ent.text)
        labels.append(ent.label_)
    return entities, labels

# https://github.com/jvfe/reconciler


def query_reconciled_wikidata(texts):
    """
    {
        "q1": {
            "query": "Hans-Eberhard Urbaniak"
        },
        "q2": {
            "query": "Ernst Schwanhold"
        }
    }
    """
    query = {'q' + str(i): {'query': text, 'limit': 3} for i, text in enumerate(texts)}
    # query = {"q0" :{"query" :"champaign"}}
    query = json.dumps(query)
    # print(query)


    tries = 0
    while tries < 3:
        try:
            response = requests.get(
                RECON_ENDPOINT, data={"queries": query}
            )
        except requests.ConnectionError:
            tries += 1
        else:
            query_result = response.json()
            if "status" in query_result and query_result["status"] == "error":
                raise requests.HTTPError(
                    "The query returned an error, check if you mistyped an argument."
                )
            else:
                return query_result
    if tries == 3:
        raise requests.ConnectionError(
            "Couldn't connect to reconciliation client")


def parse_query_result(query_result):

    if "status" in query_result and query_result["status"] == "error":
        raise requests.HTTPError(
            "The query returned an error, check if you mistyped an argument."
        )
    else:
        results = []
        for k in query_result:
            parsed_result = {}
            parsed_result['description'] = query_result[k]["result"][0]['description']
            parsed_result['type'] = query_result[k]["result"][0]['type'][0]['name']
            results.append(parsed_result)
        return results


def main():
    text = '''
    Elevator ( feat . Timbaland ) Flo Rida Mail On Sunday ( Deluxe Version )
    '''
    entities, labels = get_NER_entities(text)
    # print(entities)

    query_result = query_reconciled_wikidata(entities)
    # print(query_result)

    parsed_result = parse_query_result(query_result)
    # print(parsed_result)

    results = dict(zip(entities, parsed_result))

    print(results)

if __name__ == '__main__':
    main()
