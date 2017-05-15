# Neural Machine Translation w/Professor Forcing
This is a repository for machine translation with open license.

Working Branches
--------------
- Master: code to run the baseline
- Adversarial_stable: code to run adversarial model. 

Launching Jobs
--------------
Command:
```
THEANO_FLAGS=device=gpu,floatX=float32 python nmt.py
```
As of now no arguments to the call are implemented. To configure the model you should go to the call at the end of nmt.py and hardcode the parameters.

Translate from model
--------------
```
python translate.py [.npz filename with model] [source dict] [dest dict] [source file for translation] [destination file]
```
Example
```
 python translate.py ./saved_models/fr-en/exp1/epoch19_nbUpd630000_model.npz ../data/vocab_and_data_small_europarl_v7_enfr/vocab.fr.pkl ../data/vocab_and_data_small_europarl_v7_enfr/vocab.en.pkl ../data/small_europarl_v7_enfr_txt/valid.fr.txt ./saved_models/fr-en/exp1/epoch19_nbUpd630000_model.npz.txt
```
Compute NLL from model
--------------
Command:
```
perl evaluation/multi-bleu.perl [original file] [translated file]
```
Example:
```
perl evaluation/multi-bleu.perl ../data/small_europarl_v7_enfr_txt/valid.en.txt < ./saved_models/fr-en/exp1/epoch18_nbUpd610000_model.npz.txt 
```

Requirements
--------------

 * tables (3.3.0)
