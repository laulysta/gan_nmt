#!/bin/bash


for MODEL in $(ls $1/*.npz)
    do
        #wc $MODEL
        python translate.py $MODEL ../data/vocab_and_data_small_europarl_v7_enfr/vocab.fr.pkl ../data/vocab_and_data_small_europarl_v7_enfr/vocab.en.pkl ../data/small_europarl_v7_enfr_txt/valid.fr.txt $MODEL.txt
    done


