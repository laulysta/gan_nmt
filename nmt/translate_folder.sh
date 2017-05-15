#!/bin/bash


for MODEL in $(ls $1/*.npz)
    do
        #wc $MODEL
        python translate.py $MODEL ../data/data_vocab_europarl_en_de_h5/vocab.de.pkl ../data/data_vocab_europarl_en_de_h5/vocab.en.pkl ../data/en_de_txt/val_ende-ref.de.toc.low $MODEL.txt
    done




