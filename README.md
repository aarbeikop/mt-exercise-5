# MT Exercise 5: Byte Pair Encoding, Beam Search
This repository is a starting point for the 5th and final exercise. As before, fork this repo to your own account and the clone it into your prefered directory.

## Requirements

- This only works on a Unix-like system, with bash available.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

## Steps

Clone your fork of this repository in the desired place:

    git clone https://github.com/[your-name]/mt-exercise-5

Create a new virtualenv that uses Python 3.10. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software as described in the exercise pdf.

Download data:

    ./download_iwslt_2017_data.sh
    
Before executing any further steps, you need to make the modifications described in the exercise pdf.

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Evaluate a trained model with

    ./scripts/evaluate.sh

## How to run our Implementation

### Before Training 
    ./setup_environment.sh
    ./BPE_training.sh

### To Train
After setting up, you can run the training commands from the command line:
    python3 -m joeynmt train configs/transformer_bpe_4k.yaml
 
    python3 -m joeynmt train configs/transformer_bpe_2k.yaml
 
    python3 -m joeynmt train configs/transformer_word_2k.yaml

### For BLEU Evaluation
To evaluate, run BLEU on the test set (example call):
    sacrebleu test/test.en -i models/bpe4k/00022500.hyps.test -m bleu -w 4

### Beam Search Evaluation
To perform beam searches with different beam sizes: 
    ./beam_size_bleu.sh
