dim_word=100
dim=1000
encoder='gru'
decoder='gru_cond'
hiero=None
patience=10
max_epochs=2
dispFreq=100
decay_c=0.
alpha_c=0.
diag_c=0.
lrate=0.01
n_words_src=100000
n_words=100000
maxlen=100
optimizer='adadelta'
batch_size = 16
valid_batch_size = 16
saveto='model.npz'
validFreq=1000
saveFreq=1000
sampleFreq=100
dataset='wmt14enfr'
dictionary = '../data/small_europarl_v7_enfr_txt/vocab.fr.pkl'
dictionary_src = '../data/small_europarl_v7_enfr_txt/vocab.en.pkl'
use_dropout=False
reload_=False
correlation_coeff=0.1
clip_c=0.

model_options = {'dim_word':100,
                 'dim':1000,
                 'encoder':'gru',
                 'decoder':'gru_cond',
                 'hiero':None,
                 'patience':10,
                 'max_epochs':2,
                 'dispFreq':100,
                 'decay_c':0.,
                 'alpha_c':0.,
                 'diag_c':0.,
                 'lrate':0.01,
                 'n_words_src':100000,
                 'n_words':100000,
                 'maxlen':100,
                 'optimizer':'adadelta',
                 'batch_size' : 16,
                 'valid_batch_size' : 16,
                 'saveto':'model.npz',
                 'validFreq':1000,
                 'saveFreq':1000,
                 'sampleFreq':100,
                 'dataset':'wmt14enfr',
                 'dictionary' : '../data/small_europarl_v7_enfr_txt/vocab.fr.pkl',
                 'dictionary_src' : '../data/small_europarl_v7_enfr_txt/vocab.en.pkl',
                 'use_dropout':False,
                 'reload_':False,
                 'correlation_coeff':0.1,
                 'clip_c':0.}