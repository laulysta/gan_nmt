import numpy as np
import fcntl  # copy
import itertools
import sys, os
import argparse
import time
import datetime
from nmt import train
from os.path import join as pjoin

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dw', '--dim_word', required=False, default='50', help='Size of the word representation')
parser.add_argument('-d', '--dim_model', required=False, default='200', help='Size of the hidden representation')
parser.add_argument('-l', '--lr', required=False, default='0.001', help='learning rate')
parser.add_argument('-r', '--reload_path', required=False, default='', help='ex: pathModel.npz')
parser.add_argument('-data', '--dataset', required=False, default='testing', help='ex: testing, europarl')
parser.add_argument('-m', '--model', required=False, default='baseline', help='ex: baseline, dark')
parser.add_argument('-bs', '--batch_size', required=False, default='16', help='Size of the batch')
parser.add_argument('-out', '--out_dir', required=False, default='.', help='Output directory for the model')
parser.add_argument('-p', '--patience', required=False, default='5', help='Patience')
parser.add_argument('-lam', '--lambda', required=False, default='5', help='Lambda')

#parser.add_argument('-ec', '--euclidean_coeff', default=0.1, type=float, help='Coefficient of the Euclidean distance in the cost (if coverage vector is used).')
#parser.add_argument('-ca', '--covVec_in_attention', action="store_true", help='Coverage vector connected to the attentional part.')
#parser.add_argument('-cd', '--covVec_in_decoder', action="store_true", help='Coverage vector connected to the decoder part.')
#parser.add_argument('-cp', '--covVec_in_pred', action="store_true", help='Coverage vector connected to the prediction part.')



args = parser.parse_args()

dim_word = int(args.dim_word)
dim_model = int(args.dim_model)
lr = float(args.lr)
dataset = args.dataset
model = str(args.model)
batch_size = int(args.batch_size)
reload_path = args.reload_path
patience = int(args.patience)
lambda_adv = int(args.lambda)

list_options = [str(model), str(dim_word), str(dim_model), str(lr), str(batch_size), str(patience), str(lambda_adv)]

#Create names and folders
####################################################################################
if reload_path == '':
    dirPath = pjoin(args.out_dir, 'saved_models_' + model)
    if not os.path.exists(dirPath):
        try:
            os.makedirs(dirPath)
        except OSError as e:
            print e
            print 'Exeption was catch, will continue script \n'


    str_options = "_".join(list_options)
    if dataset == "testing":
        dirModelName = "model_gru_testing_ende_" + str_options
    elif dataset == "europarl_en_de":
        dirModelName = "model_gru_europarl_ende_" + str_options
    elif dataset == "beam_europarl_en_de":
        dirModelName = "model_gru_beam_europarl_ende_" + str_options
    else:
        sys.exit("Wrong dataset")

    dirPathModel = pjoin(dirPath, dirModelName)
    if not os.path.exists(dirPathModel):
        try:
            os.makedirs(dirPathModel)
        except OSError as e:
            print e
            print 'Exeption was catch, will continue script \n'

    modelName = os.path.join(dirPathModel, dirModelName + ".npz")
    #modelName = dirModelName + ".npz"

    dirPathOutput = pjoin(dirPathModel, 'output')
    if not os.path.exists(dirPathOutput):
        try:
            os.makedirs(dirPathOutput)
        except OSError as e:
            print e
            print 'Exeption was catch, will continue script \n'

###################################################################################


if dataset == "testing":
    n_words_src = 100#3449
    n_words_trg = 100#4667
    datasets=['../data/dataset_testing/trainset_en.txt', 
              '../data/dataset_testing/trainset_de.txt']
    valid_datasets=['../data/dataset_testing/validset_en.txt', 
                    '../data/dataset_testing/validset_de.txt',
                    '../data/dataset_testing/validset_de.txt']
    other_datasets=['../data/dataset_testing/testset_en.txt', 
                    '../data/dataset_testing/testset_de.txt',
                    '../data/dataset_testing/testset_de.txt']
    dictionaries=['../data/dataset_testing/vocab_en.pkl', 
                  '../data/dataset_testing/vocab_de.pkl']

    sizeTrainset = 1000.0
    #batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)


elif dataset == "europarl_en_de":
    n_words_src=20000
    n_words_trg=20000
    datasets=['../data/europarl_de-en_txt.tok.low/europarl-v7.de-en.en.toc.low', 
              '../data/europarl_de-en_txt.tok.low/europarl-v7.de-en.de.toc.low']
    valid_datasets=['../data/europarl_de-en_txt.tok.low/newstest2015-ende-src.en.toc.low', 
                    '../data/europarl_de-en_txt.tok.low/newstest2015-ende-ref.de.toc.low',
                    '../data/europarl_de-en_txt.tok.low/newstest2015-ende-ref.de.toc.low']
    other_datasets=['../data/europarl_de-en_txt.tok.low/newstest2016-ende-src.en.toc.low', 
                    '../data/europarl_de-en_txt.tok.low/newstest2016-ende-ref.de.toc.low',
                    '../data/europarl_de-en_txt.tok.low/newstest2016-ende-ref.de.toc.low']
    dictionaries=['../data/europarl_de-en_txt.tok.low/vocab_en.pkl', 
                  '../data/europarl_de-en_txt.tok.low/vocab_de.pkl']

    sizeTrainset = 1920210.0
    #batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)

elif dataset == "beam_europarl_en_de":
    n_words_src=20000
    n_words_trg=20000
    datasets=['../data/beam_europarl_de-en_txt.tok.low/big_europarl-v7.de-en.en.toc.low', 
              '../data/beam_europarl_de-en_txt.tok.low/big_europarl-v7.de-en.de.toc.low']
    valid_datasets=['../data/beam_europarl_de-en_txt.tok.low/newstest2015-ende-src.en.toc.low', 
                    '../data/beam_europarl_de-en_txt.tok.low/newstest2015-ende-ref.de.toc.low',
                    '../data/beam_europarl_de-en_txt.tok.low/newstest2015-ende-ref.de.toc.low']
    other_datasets=['../data/beam_europarl_de-en_txt.tok.low/newstest2016-ende-src.en.toc.low', 
                    '../data/beam_europarl_de-en_txt.tok.low/newstest2016-ende-ref.de.toc.low',
                    '../data/beam_europarl_de-en_txt.tok.low/newstest2016-ende-ref.de.toc.low']
    dictionaries=['../data/beam_europarl_de-en_txt.tok.low/vocab_en.pkl', 
                  '../data/beam_europarl_de-en_txt.tok.low/vocab_de.pkl']

    sizeTrainset = 1536167.0
    #batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)

if reload_path != '':
    reload_ = True
    modelName = reload_path
    dirModelName = modelName.split('/')[-1] + '_reload'
    dirPath = '/'.join(modelName.split('/')[0:-1])
    dirPathOutput = pjoin(dirPath, 'output')
    if not os.path.exists(dirPathOutput):
        try:
            os.makedirs(dirPathOutput)
        except OSError as e:
            print e
            print 'Exeption was catch, will continue script \n'
else:
    reload_ = False
saveFreq = nb_batch_epoch
use_dropout = True
if model == 'baseline':
    decoder = 'gru_cond_legacy'
elif model == 'dark':
    decoder = 'gru_cond_legacy_dark'
else:
    sys.exit("Wrong arg model")


trainerr, validerr, testerr = train(saveto=modelName,
                                    reload_=reload_,
                                    dim_word=dim_word,
                                    dim=dim_model,
                                    encoder='gru',
                                    decoder='gru_cond', # 'gru_cond_legacy_lbc', # if args.covVec_in_attention or args.covVec_in_decoder else 'gru_cond',
                                    max_epochs=100,
                                    n_words_src=n_words_src,
                                    n_words=n_words_trg,
                                    optimizer='adadelta',
                                    decay_c=0.,
                                    alpha_c=0.,
                                    clip_c=1.,
                                    lrate=lr,
                                    patience=patience,
                                    maxlen=50,
                                    batch_size=batch_size,
                                    valid_batch_size=batch_size,
                                    validFreq=nb_batch_epoch, # freq in batch of computing cost for train, valid and test
                                    dispFreq=nb_batch_epoch, # freq of diplaying the cost of one batch (e.g.: 1 is diplaying the cost of each batch)
                                    saveFreq=saveFreq, # freq of saving the model per batch
                                    sampleFreq=nb_batch_epoch, # freq of sampling per batch
                                    datasets=datasets,
                                    valid_datasets=valid_datasets,
                                    other_datasets=other_datasets,
                                    dictionaries=dictionaries,
                                    use_dropout=use_dropout,
                                    adversarial_mode='simple',
                                    adversarial_cost='default',
                                    lambda_adv=lambda_adv)

# Prepare result line to append to result file
line = "\t".join([str(dirModelName), str(dataset)] + list_options + [str(nb_epoch), str(nb_batch), str(validerr), str(testerr), str(validbleu), str(testbleu)]) + "\n"

# Preparing result file
results_file = dirPath + '/results.txt'
if not os.path.exists(results_file):
    # Create result file if doesn't exist
    header_line = "\t".join(['dirModelName', 'dataset', 'model', 'dim_word', 'dim_model', 'lr', 'batch_size',
                             'nb_epoch', 'nb_batch', 'valid_err', 'test_err', 'valid_BLEU', 'test_BLEU']) + '\n'
    f = open(results_file, 'w')
    f.write(header_line)
    f.close()

f = open(results_file, "a")
fcntl.flock(f.fileno(), fcntl.LOCK_EX)
f.write(line)
f.close()  # unlocks the file
