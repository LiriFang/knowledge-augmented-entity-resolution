# T-KAER: Transparency-enabled Knowledge-Augmented Entity Resolution Framework
## Environment Setup 
First, install dependencies: 
```
conda env create --name kaer_39 --file=environments.yml
```

Second, install [Refined](https://github.com/amazon-science/ReFinED) for entity linking:
```
pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip
```
## Experiments: 
- [Experiment I]: Run KAER and Documenting Experimental Process
- [Experiment II]: Evaluating and Analyzing ER results  

#### Experiment I: Run KAER and Documenting Experimental Process

1. Commands and HyperParameters

Entity Resolution by Pre-trained Language Models (PLMs) can be started by running `train_ditto.py` script under `dittoPlus` folder.

The command and key hyperparameters can be tuned by users are as follows:

```
$ cd dittoPlus
$ python train_ditto.py --task {*} --dk {*} --prompt {*} --kbert {*}
```

* `task`: dataset folder name (trainset, validset, and testset), all meta-information documented in `dittoPlus/configs.json`. 
* `dk`: domain knowledge name: {default:none (ditto baseline), sherlock, doduo, entityLinking}
* `prompt`: prompting methods name: {default: 1 (space), 0: kbert, 2 (slash)}
* `kbert`: using kbert (constrained pruning method) or not: {default: False, True}

2. Experiment Result: Log File Generated

After the experiment, one log file will be generated and can be found under this directory: `dittoPlus/output/`.

#### Experiment II: Evaluating and Analyzing ER results  
1. Evaluating script based on the log files: `dittoPlus/ev_results.py`
2. Compare the performance across the KA methods
## Directory and Descriptions 
| Directory | Contents Descriptions |
| ----------- | ----------- |
| data | Dataset from The ER-Magellan Benchmark |
| environment.yml | All Dependencies Required to Run the Experiments {sherlock} |
| dittoPlus | ditto + Domain Knowledge |


## Related Papers
[1] Fang, L., Li, L., Liu, Y., Torvik, V. I., & Ludäscher, B. (2023). KAER: A Knowledge Augmented Pre-Trained Language Model for Entity Resolution. arXiv preprint arXiv:2301.04770.
[2] Li, L., Fang, L., Liu, Y., Torvik, V. I., & Ludäscher, B. (2024). T-KAER: Transparency-enhanced Pre-Trained Language Model for Entity Resolution. IDCC, 18.



