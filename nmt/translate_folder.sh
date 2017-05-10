#!/bin/bash


for MODEL in $(ls $1/*.npz)
    do
        #wc $MODEL
        python translate.py $MODEL ../data/data_vocab_europarl_en_de_h5/vocab.de.pkl ../data/data_vocab_europarl_en_de_h5/vocab.en.pkl ../data/en_de_txt/plus50_3k_europarl-v7.de-en.de.toc.low $MODEL.txt
    done
