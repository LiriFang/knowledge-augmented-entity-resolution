import gc
import time

import os
import argparse
import json
import sys
import torch
import numpy as np
import random
import jsonlines

from scipy.special import softmax
from torch.utils import data

from tensorflow.keras import backend as K
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train


def classify(sentence_pairs, model, save,
             lm='distilbert',
             max_len=256,
             threshold=None):
    """Apply the MRPC model.

    Args:
        sentence_pairs (list of str): the sequence pairs
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length
        threshold (float, optional): the threshold of the 0's class

    Returns:
        list of float: the scores of the pairs
    """
    enc = []
    inputs = sentence_pairs
    # print('max_len =', max_len)
   
    dataset = DittoDataset(inputs,
                           max_len=max_len,
                           lm=lm)
    padder = dataset.pad
    # print(dataset[0])
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=len(dataset),
                               shuffle=False,
                               num_workers=0,
                               collate_fn=padder
                               )
                            #    collate_fn=DittoDataset.pad)

    # prediction
    all_probs = []
    all_logits = []
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            if len(batch) == 2:
                x, y = batch
                logits = model(x, save=save)
            elif len(batch) == 4:
                x, position_batch, visible_matrix_batch, y = batch
                # visible_matrix_batch, position_batch = visible_matrix_batch.to(model.device), position_batch.to(model.device)
                logits = model(x, vm=visible_matrix_batch, position_ids=position_batch, save=save)
                del position_batch, visible_matrix_batch
            enc.append(model.enc)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_logits += logits.cpu().numpy().tolist()
        enc = np.concatenate(enc)
    if threshold is None:
        threshold = 0.5

    pred = [1 if p > threshold else 0 for p in all_probs]
    print(f'dimension of enc: {enc.size}')
    return pred, all_logits, enc


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/iTunes-Amazon")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--prompt", type=int, default=1)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--device", type=str, default='cuda', help='cpu or cuda')
    parser.add_argument("--kbert",type=bool, default=False)
    parser.add_argument("--overwrite",type=bool, default=False)

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset_input = config['trainset']
    validset_input = config['validset']
    testset_input = config['testset']

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset_input = summarizer.transform_file(trainset_input, max_len=hp.max_len, overwrite=True)
        validset_input = summarizer.transform_file(validset_input, max_len=hp.max_len, overwrite=True)
        testset_input = summarizer.transform_file(testset_input, max_len=hp.max_len, overwrite=True)
    
    # out_fn = input_fn + f'.prompt_type{prompt_type}.sherlock.dk'
    if hp.dk == 'sherlock':
        trainset = trainset_input + f'.prompt_type{hp.prompt}.sherlock.dk'
        testset = testset_input + f'.prompt_type{hp.prompt}.sherlock.dk'
        validset = validset_input + f'.prompt_type{hp.prompt}.sherlock.dk'
    # TODO: what's the extension for EL- file?

    
    logging_info = {
    'dataset-path': testset,
    'hyperparams': {
        'prompt': hp.prompt,
        'batch_size': hp.batch_size,
        'max_len': hp.max_len,
        'lr': hp.lr,
        'kbert': hp.kbert

    },
    'rows':[
    ],
    # 'ground_truth':,
    # 'pred_result':,
    # 'matching_conf':,
    }
    # row: {'left': ..., 'right':..., 'ground_truth':0, 'pred_result':0, 'matching_conf':...}
    if os.path.exists(trainset):
        print(f"The file '{trainset}' exists already.")
    else:
        print(f"The file '{trainset}' does not exist.")
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        if hp.dk == 'entityLinking':
            injector = EntityLinkingDKInjector(config, hp.dk)
        if hp.dk == 'sherlock':
            injector = SherlockDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset= injector.transform_file(trainset_input, trainset, overwrite=hp.overwrite,prompt_type=hp.prompt)
        validset= injector.transform_file(validset_input, validset, overwrite=hp.overwrite,prompt_type=hp.prompt)
        testset= injector.transform_file(testset_input, testset, overwrite=hp.overwrite,prompt_type=hp.prompt)
    
    #load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da,
                                   kbert=hp.kbert)
    valid_dataset = DittoDataset(validset, lm=hp.lm, max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da,
                                   kbert=hp.kbert)
    test_dataset = DittoDataset(testset, lm=hp.lm, max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da,
                                   kbert=hp.kbert)

    # train and evaluate the model
    model = train(train_dataset,
          valid_dataset,
          test_dataset,
          run_tag, hp)

    # predict the model
    # batch processing
    def process_batch(rows, pairs, save , writer, logs):
        predictions, logits, vectors = classify(rows, model, save=save, lm=hp.lm,
                                        max_len=hp.max_len,
                                        threshold=0.5)
        print(len(vectors))
        assert len(rows) == len(vectors)
        scores = softmax(logits, axis=1)
        for idx, (pair, pred, score) in enumerate(zip(pairs, predictions, scores)):
            row_v = rows[idx]
            s1, s2, label = row_v.strip().split('\t')
            output = {'left': pair[0], 'right': pair[1],
                'match': pred,
                'match_confidence': score[int(pred)]}
            row = {
                'row_index': idx,
                'left': pair[0], 'right': pair[1], 
                'vectors': vectors[idx],
                'ground_truth':label,
                'pred_result': pred,
                'match_confidence': score[int(pred)]}
            logs.append(row)
            writer.write(output)
    
    start_time = time.time()
    os.makedirs(f'./output/{hp.task}', exist_ok=True)
    save = True
    with jsonlines.open(f"./output/{hp.task}/result.jsonl", mode='w') as writer:
        pairs = test_dataset.pairs # (e1, e2)
        rows = test_dataset.rows # (e1, e2, \t, label)
        if len(pairs) == hp.batch_size:
            process_batch(rows, pairs, save, writer, logging_info['rows'])
            pairs.clear()
            rows.clear()

        if len(pairs) > 0:
            process_batch(rows, pairs, save, writer, logging_info['rows'])

    run_time = time.time() - start_time
    run_tag = '%s_lm=%s_dk=%s_su=%s' % (config['name'], hp.lm, str(hp.dk != None), str(hp.summarize != None))
    os.system('echo %s %f >> log.txt' % (run_tag, run_time))
    print(logging_info)
    