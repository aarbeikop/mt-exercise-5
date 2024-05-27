#!/bin/bash

# Clone necessary repositories
git clone https://github.com/EdinburghNLP/nematus
git clone https://github.com/marpng/mt-exercise-5

# Make the virtual environment
./scripts/make_virtualenv.sh
source ./scripts/../venvs/torch3/bin/activate

# Set permissions
chmod 755 mt-exercise-5

# Install required Python packages
pip install subword-nmt
pip install sacremoses

# Clone and install JoeyNMT
git clone https://github.com/marpng/joeynmt
cd joeynmt && pip install -e .
cd ..

# Download IWSLT 2017 data
scripts/download_iwslt_2017_data.sh

# Create training data directory
mkdir train_data

# Limit data to 100,000 lines
head -n100000 data/train.en-nl.en >> train_data/small_train.en-nl.en
head -n100000 data/train.en-nl.nl >> train_data/small_train.en-nl.nl

# Tokenize training data
mkdir train
sacremoses -l en -j 4 tokenize < train_data/small_train.en-nl.en > train/train.trg
sacremoses -l nl -j 4 tokenize < train_data/small_train.en-nl.nl > train/train.src

# Create dev and test directories and move data
mkdir dev
mkdir test
mv data/dev.nl-en.en dev/dev.trg
mv data/dev.nl-en.nl dev/dev.src
mv data/test.nl-en.en test/test.trg
mv data/test.nl-en.nl test/test.src
