#!/bin/bash

# Create directory for BPE files
mkdir -p bpe

# Learn BPE Codes
subword-nmt learn-joint-bpe-and-vocab --input train/train.nl train/train.en -s 2000 -o bpe/codes2k.bpe --write-vocabulary bpe/vocab_2k.nl bpe/vocab_2k.en
subword-nmt learn-joint-bpe-and-vocab --input train/train.nl train/train.en -s 4000 -o bpe/codes4k.bpe --write-vocabulary bpe/vocab_4k.nl bpe/vocab_4k.en

# Apply BPE Codes to My Data
subword-nmt apply-bpe -c bpe/codes2k.bpe --vocabulary bpe/vocab_2k.nl --vocabulary-threshold 5 < train/train.nl > bpe/train2k.BPE.nl
subword-nmt apply-bpe -c bpe/codes2k.bpe --vocabulary bpe/vocab_2k.en --vocabulary-threshold 5 < train/train.en > bpe/train2k.BPE.en
subword-nmt apply-bpe -c bpe/codes4k.bpe --vocabulary bpe/vocab_4k.nl --vocabulary-threshold 5 < train/train.nl > bpe/train4k.BPE.nl
subword-nmt apply-bpe -c bpe/codes4k.bpe --vocabulary bpe/vocab_4k.en --vocabulary-threshold 5 < train/train.en > bpe/train4k.BPE.en

# Generate BPE Vocabulary
cat bpe/train2k.BPE.nl bpe/train2k.BPE.en | subword-nmt get-vocab > bpe/vocab2k.bpe
cat bpe/train4k.BPE.nl bpe/train4k.BPE.en | subword-nmt get-vocab > bpe/vocab4k.bpe

# Process BPE vocabulary files by extracting only the tokens, discarding frequency counts
# Creates new 'clean' vocabulary files and removes the original files to maintain cleanliness
cut -d' ' -f1 bpe/vocab2k.bpe > bpe/clean_vocab_2k.nl
cut -d' ' -f1 bpe/vocab2k.bpe > bpe/clean_vocab_2k.en
rm bpe/vocab2k.bpe

cut -d' ' -f1 bpe/vocab4k.bpe > bpe/clean_vocab_4k.nl
cut -d' ' -f1 bpe/vocab4k.bpe > bpe/clean_vocab_4k.en
rm bpe/vocab4k.bpe

# Generate Vocabulary Files for each language
python nematus/data/build_dictionary.py bpe/train2k.BPE.en
python nematus/data/build_dictionary.py bpe/train2k.BPE.nl
python nematus/data/build_dictionary.py bpe/train4k.BPE.en
python nematus/data/build_dictionary.py bpe/train4k.BPE.nl
