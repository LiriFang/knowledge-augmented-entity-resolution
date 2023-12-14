<<<<<<< HEAD
import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/iTunes-Amazon")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)

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

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len, overwrite=True)
        validset = summarizer.transform_file(validset, max_len=hp.max_len, overwrite=True)
        testset = summarizer.transform_file(testset, max_len=hp.max_len, overwrite=True)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        if hp.dk == 'entityLinking':
            injector = EntityLinkingDKInjector(config, hp.dk)
        if hp.dk == 'sherlock':
            injector = SherlockDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset, overwrite=True)
        validset = injector.transform_file(validset, overwrite=True)
        testset = injector.transform_file(testset, overwrite=True)

    # load train/dev/test sets
    # TODO： required recording 
    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)

    # train and evaluate the model
    train(train_dataset,
          valid_dataset,
          test_dataset,
          run_tag, hp)
=======
import gc
import time

import os
import argparse
import json
import sys
import torch
import numpy as np
import random

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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Dirty/DBLP-GoogleScholar")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
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
    parser.add_argument("--prompt", type=int, default=0)
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

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    if hp.dk == 'doduo':
        trainset = trainset+'.doduo'
        testset = testset + '.doduo'
        validset = validset+ '.doduo'

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len, overwrite=hp.overwrite)
        validset = summarizer.transform_file(validset, max_len=hp.max_len, overwrite=hp.overwrite)
        testset = summarizer.transform_file(testset, max_len=hp.max_len, overwrite=hp.overwrite)


    # if hp.ct is not None:
    #     pass

    if hp.dk is not None and hp.dk != "doduo":
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        elif hp.dk == 'entityLinking':
            injector = EntityLinkingDKInjector(config, hp.dk)
        elif hp.dk == 'sherlock':
            injector = SherlockDKInjector(config, hp.dk)
            #todo connect sherlock with k-bert
        else:
            injector = GeneralDKInjector(config, hp.dk)

    trainset= injector.transform_file(trainset, overwrite=hp.overwrite, fname=f"train_{config['name']}",prompt_type=hp.prompt)
    validset= injector.transform_file(validset, overwrite=hp.overwrite,fname=f"valid_{config['name']}",prompt_type=hp.prompt)
    testset= injector.transform_file(testset, overwrite=hp.overwrite,fname=f"test_{config['name']}",prompt_type=hp.prompt)
    
    # add visible matrix by K-bert



    # load train/dev/test sets
    # print(hp.kbert)
    # raise NotImplementedError
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
    train(train_dataset,
          valid_dataset,
          test_dataset,
          run_tag, hp)
>>>>>>> main
