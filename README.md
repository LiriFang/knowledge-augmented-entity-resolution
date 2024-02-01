# data-preparator-for-EM
## Environment Setup 
First, install dependencies: 
```
conda env create --name kaer_39 --file=environments.yml
```

Second, install [Refined](https://github.com/amazon-science/ReFinED) for entity linking:
```
pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip
```
## Commands and HyperParameters
Entity Resolution by Pre-trained Language Models (PLMs) can be started by running train_ditto.py script under `dittoPlus` folder.

The command and key hyperparameters can be tuned by users are as follows:

```
$ cd dittoPlus
$ python train_ditto.py --task {Dataset name} --dk {knowledge augmentation methods} --prompt {0,1,2} --kbert {True, False}
```

* `task`: dataset folder name (trainset, validset, and testset), all meta-information documented in `dittoPlus/configs.json`. 
* `dk`: domain knowledge name: {default:none (ditto baseline), sherlock, doduo, entityLinking}
* `prompt`: prompting methods name: {default: 1 (space), 0: kbert, 2 (slash)}
* `kbert`: using kbert (constrained pruning method) or not: {default: False, True}

## Running Experiments 


## Directory and Descriptions 
| Directory | Contents Descriptions |
| ----------- | ----------- |
| data | Dataset from The ER-Magellan Benchmark |
| dittoPlus | Text |
| environment.yml | All Dependencies Required to Run the Experiments {sherlock} |
| dittoPlus | Text |


## Related Papers




