'''
Build a attention-based neural machine translation model
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

import wmt14enfr
import iwslt14zhen
import openmt15zhen
import trans_enhi
import stan

import inspect

from utils import *
from layers import *
from optimizers import *

from theano.gradient import disconnected_grad
from theano.compile.nanguardmode import NanGuardMode

theano.config.floatX = 'float32'
TINY = tensor.alloc(1e-6).astype('float32')
#theano.config.dnn.enabled = False
# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
datasets = {'wmt14enfr': (wmt14enfr.load_data, wmt14enfr.prepare_data),
            'iwslt14zhen': (iwslt14zhen.load_data, iwslt14zhen.prepare_data),
            'openmt15zhen': (openmt15zhen.load_data, openmt15zhen.prepare_data),
            'trans_enhi': (trans_enhi.load_data, trans_enhi.prepare_data),
            'stan': (stan.load_data, stan.prepare_data)
            }


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def init_params_nll(tparams):
    params_nll = OrderedDict()
    for kk, pp in tparams.iteritems():
        if not 'adversarial' in kk and not 'FR' in kk:
            params_nll[kk] = tparams[kk]
    return params_nll

def init_params_adversarial(tparams):
    params_adversarial = OrderedDict()
    for kk, pp in tparams.iteritems():
        if 'adversarial' in kk:
            params_adversarial[kk] = tparams[kk]
    return params_adversarial

def init_params_gen_adversarial(tparams):
    disconnected_params = ['decoder_W_comb_att', 'decoder_U_att',
                            'decoder_c_tt', 'decoder_Wc_att',
                            'decoder_b_att', 'Wemb',
                            'Wemb_dec',
                            'encoder_W',
                            'encoder_b',
                            'encoder_U',
                            'encoder_Wx',
                            'encoder_Ux',
                            'encoder_bx',
                            'encoder_r_W',
                            'encoder_r_b',
                            'encoder_r_U',
                            'encoder_r_Wx',
                            'encoder_r_Ux',
                            'encoder_r_bx',
                            'ff_state_w',
                            'ff_state_b']

    params_adversarial = OrderedDict()
    for kk, pp in tparams.iteritems():
        if kk not in disconnected_params and not 'ff_logit' in kk and not 'ff_nb' in kk:
            if not 'adversarial' in kk:
                params_adversarial[kk] = tparams[kk]
    return params_adversarial

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('{} is not in the archive'.format(kk))
            continue
        params[kk] = pp[kk]

    return params


# initialize all parameters
def init_params(options):
    numpy.random.seed(1234)
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # Encoder
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])

    ctxdim = options['dim']
    if not options['decoder'].endswith('simple'):
        ctxdim = options['dim'] * 2
        params = get_layer(options['encoder'])[0](options, params, prefix='encoder_r',
                                                  nin=options['dim_word'], dim=options['dim'])
        if options['hiero']:
            params = get_layer(options['hiero'])[0](options, params, prefix='hiero',
                                                    nin=2 * options['dim'],
                                                    dimctx=2 * options['dim'])
    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=ctxdim, nout=options['dim'])
    if options['encoder'] == 'lstm':
        params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=ctxdim, nout=options['dim'])

    # decoder: Teacher Forcing Mode
    params = get_layer(options['decoder'])[0](options, params, prefix='decoder',
                                              nin=options['dim_word'], dim=options['dim'],
                                              dimctx=ctxdim)

    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_word'], ortho=False)
    params = get_layer('ff_nb')[0](options, params, prefix='ff_nb_logit_prev', nin=options['dim_word'], nout=options['dim_word'], ortho=False)
    params = get_layer('ff_nb')[0](options, params, prefix='ff_nb_logit_ctx', nin=ctxdim, nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], nout=options['n_words'])

    # decoder: Free Running Mode
    # params = get_layer(options['decoder_FR'])[0](options, params, prefix='decoder_FR',
    #                                               nin=options['dim_word'], dim=options['dim'],
    #                                               dimctx=ctxdim)

    #Adversarial network
    params = get_layer('gru')[0](options, params, prefix='encoder_adversarial',
                                       nin=options['dim'], dim=options['dim'])

    params = get_layer('mlp_adversarial')[0](options, params, prefix='mlp_adversarial',
                                       nin=options['dim'] * 2, dim=options['dim'] * 2)
    return params

def build_discriminator_adversarial(B_orig, B_fake, tparams, options):
    trng = RandomStreams(1234)

    # description string: #hidden_states x #samples
    #B_orig = tensor.matrix('B_orig', dtype='float32')
    #B_fake = tensor.matrix('B_fake', dtype='float32')
    #h_orig_mask = tensor.matrix('h_orig_mask', dtype='float32')
    #h_fake_mask = tensor.matrix('h_fake_mask', dtype='float32')

    B_orig_r = B_orig[::-1]
    B_fake_r = B_fake[::-1]
    #h_orig_r_mask = h_orig_mask[::-1]
    #h_fake_r_mask = h_fake_mask[::-1]

    n_timesteps_orig = B_orig.shape[0]
    n_timesteps_fake = B_fake.shape[0]

    # RNN for adversarial network
    encoder = get_layer(options['encoder'])[1]
    proj_orig = encoder(tparams, B_orig, options, prefix='encoder_adversarial')
    proj_fake = encoder(tparams, B_fake, options, prefix='encoder_adversarial')
    proj_orig = proj_orig[0]
    proj_fake = proj_fake[0]

    proj_orig_r = encoder(tparams, B_orig_r, options, prefix='encoder_adversarial')
    proj_fake_r = encoder(tparams, B_fake_r, options, prefix='encoder_adversarial')
    proj_orig_r = proj_orig_r[0]
    proj_fake_r = proj_fake_r[0]

    ctx_mean_orig = concatenate([proj_orig[-1], proj_orig_r[-1]], axis=proj_orig.ndim - 2)
    ctx_mean_fake = concatenate([proj_fake[-1], proj_fake_r[-1]], axis=proj_orig.ndim - 2)

    D_orig = concatenate([proj_orig, proj_orig_r[::-1]], axis=proj_orig.ndim - 1)
    D_fake = concatenate([proj_fake, proj_fake_r[::-1]], axis=proj_fake.ndim - 1)
    #D_orig =

    #mlp_adversarial = get_layer('mlp_adversarial')[1]

    D_orig = mlp_layer(tparams, ctx_mean_orig, options, prefix='mlp_adversarial')
    D_fake = mlp_layer(tparams, ctx_mean_fake, options, prefix='mlp_adversarial')

    # inps = [B_orig, B_fake]
    # outs = [D_orig, D_fake]

    # discriminator_adversarial = theano.function(inps, outs, name='discriminator_adversarial', profile=profile)

    # return discriminator_adversarial
    return D_orig, D_fake

def build_adversarial_discriminator_cost(D_orig, D_fake, tparams, options):
    #D_orig = tensor.matrix('D_orig', dtype='float32')
    #D_fake = tensor.matrix('D_fake', dtype='float32')
    
    # Review
    cost = -tensor.mean(tensor.log(1e-6 + D_orig) + tensor.log(1e-6 + 1. - D_fake))
    inps = [D_orig, D_fake]
    outs = [cost]

    #discriminator_adversarial_cost = theano.function(inps, outs, name='discriminator_adversarial_cost', profile=profile)
    #return discriminator_adversarial_cost
    return cost

def build_adversarial_generator_cost(D_fake,tparams, options):
    #D_fake = tensor.matrix('D_fake', dtype='float32')
    cost = -tensor.mean(tensor.log(D_fake + 1e-6))

    #adversarial_generator_cost = theano.function([D_fake], [cost], name='adversarial_generator_cost', profile=profile)
    #return adversarial_generator_cost
    return cost


def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]
    # src_lengths = x_mask.sum(axis=0)

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    encoder = get_layer(options['encoder'])[1]

    proj = encoder(tparams, emb, options, prefix='encoder', mask=x_mask)

    if options['decoder'].endswith('simple'):
        ctx = proj[0][-1]
        ctx_mean = ctx
    else:
        embr = tparams['Wemb'][xr.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
        encoder_r = get_layer(options['encoder'])[1]
        projr = encoder_r(tparams, embr, options, prefix='encoder_r', mask=xr_mask)
        ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)
        if options['hiero']:
            # ctx = tensor.dot(ctx, tparams['W_hiero'])
            rval = get_layer(options['hiero'])[1](tparams, ctx, options,
                                                  prefix='hiero',
                                                  context_mask=x_mask)
            ctx = rval[0]
            opt_ret['hiero_alphas'] = rval[2]
            opt_ret['hiero_betas'] = rval[3]
        # initial state/cell
        # ctx_mean = ctx.mean(0)
        # ctx_mean = (ctx * x_mask[:,:,None]).sum(0) / x_mask.sum(0)[:,None]
        # Get the last hidden states from the direct and reversed encoding
        ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim - 2)

    ff_encoder_decoder = get_layer('ff')[1]
    init_state = ff_encoder_decoder(tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = None

    if options['encoder'] == 'lstm':
        init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

    # word embedding (target)
    emb = tparams['Wemb_dec'][y.flatten()].reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # Decoder in Teacher Forcing mode
    decoder = get_layer(options['decoder'])[1]
    proj = decoder(tparams, emb, options, prefix='decoder', mask=y_mask,
                   context=ctx, context_mask=x_mask, one_step=False,
                   init_state=init_state, init_memory=init_memory)
    proj_h = proj[0]

    if options['decoder'].endswith('simple'):
        ctxs = ctx[None, :, :]
    elif options['decoder'].startswith('lstm'):
        ctxs = proj[2]
        opt_ret['dec_alphas'] = proj[3]
    else:
        ctxs = proj[1]
        opt_ret['dec_alphas'] = proj[2]

    B_teacher_forcing = proj[3]

    # Decoder in Free Running mode
    decoder_FR = get_layer(options['decoder_FR'])[1]
    proj_FR = decoder_FR(tparams, emb, options, prefix='decoder', mask=y_mask,
                      context=ctx, context_mask=x_mask, one_step=False,
                      init_state=init_state, init_memory=init_memory)
    proj_h_FR = proj_FR[0]

    if options['decoder'].endswith('simple'):
        ctxs_FR = ctx[None, :, :]
    elif options['decoder'].startswith('lstm'):
        ctxs_FR = proj_FR[2]
        opt_ret['dec_alphas'] = proj_FR[3]
    else:
        ctxs_FR = proj_FR[1]
        opt_ret['dec_alphas_FR'] = proj_FR[2]

    B_free_running = proj_FR[3]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff_nb')[1](tparams, emb, options, prefix='ff_nb_logit_prev', activ='linear')
    logit_ctx = get_layer('ff_nb')[1](tparams, ctxs, options, prefix='ff_nb_logit_ctx', activ='linear')

    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')

    logit_shp = logit.shape

    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat

    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    # Adversarial step
    #D_adversarial = build_discriminator_adversarial(tparams, options)
    D_orig, D_fake = build_discriminator_adversarial(B_teacher_forcing, B_free_running, tparams, options)
    #D_orig, D_fake = D_adversarial(B_teacher_forcing, B_free_running)
    # inps = [B_orig, B_fake]
    # outs = [D_orig, D_fake]
    #copute_cost_discriminator = build_adversarial_discriminator_cost(tparams, options)
    cost_discriminator = build_adversarial_discriminator_cost(D_orig, D_fake, tparams, options)
    # inps = [D_orig, D_fake]
    # outs = [cost]
    #cost_discriminator = compute_cost_discriminator(D_orig, D_fake)

    # compute_cost_generator = build_adversarial_generator_cost(tparams, options)
    # cost_generator = compute_cost_generator(D_fake)
    cost_generator = build_adversarial_generator_cost(D_fake,tparams, options)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, cost_discriminator, cost_generator, B_teacher_forcing, B_free_running, D_orig, D_fake


# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options, prefix='encoder')

    if options['decoder'].endswith('simple'):
        ctx = proj[0][-1]
        ctx_mean = ctx
    else:
        projr = get_layer(options['encoder'])[1](tparams, embr, options, prefix='encoder_r')
        ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)
        if options['hiero']:
            rval = get_layer(options['hiero'])[1](tparams, ctx, options, prefix='hiero')
            ctx = rval[0]
        # initial state/cell
        # ctx_mean = ctx.mean(0)
        ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim - 2)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')

    if options['encoder'] == 'lstm':
        init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    if options['decoder'].startswith('lstm'):
        outs += [init_memory]

    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')

    init_state = tensor.matrix('init_state', dtype='float32')

    if options['decoder'].startswith('lstm'):
        init_memory = tensor.matrix('init_memory', dtype='float32')
    else:
        init_memory = None

    n_timesteps = ctx.shape[0]

    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:, None] < 0, tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state,
                                            init_memory=init_memory)

    if options['decoder'].endswith('simple'):
        next_state = proj
        ctxs = ctx
    else:
        next_state = proj[0]
        ctxs = proj[1]
        if options['decoder'].startswith('lstm'):
            next_memory = proj[1]
            ctxs = proj[2]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options, prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff_nb')[1](tparams, emb, options, prefix='ff_nb_logit_prev', activ='linear')
    logit_ctx = get_layer('ff_nb')[1](tparams, ctxs, options, prefix='ff_nb_logit_ctx', activ='linear')

    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)

    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    if options['decoder'].startswith('lstm'):
        inps += [init_memory]
        outs += [next_memory]

    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


# generate sample
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30, minlen=-1, stochastic=True, argmax=False):
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if options['decoder'].startswith('lstm'):
        hyp_memories = []

    ret = f_init(x)
    next_state = ret.pop(0)
    ctx0 = ret.pop(0)
    if options['decoder'].startswith('lstm'):
        next_memory = ret.pop(0)

    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        if options['decoder'].endswith('simple'):
            ctx = numpy.tile(ctx0, [live_k, 1])
        else:
            ctx = numpy.tile(ctx0.reshape((ctx0.shape[0],ctx0.shape[2])), 
                                          [live_k, 1, 1]).transpose((1,0,2))
        inps = [next_w, ctx, next_state]
        if options['decoder'].startswith('lstm'):
            inps += [next_memory]

        ret = f_next(*inps)
        next_p = ret.pop(0)
        next_w = ret.pop(0)
        next_state = ret.pop(0)
        if options['decoder'].startswith('lstm'):
            next_memory = ret.pop(0)

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0,nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            if options['decoder'].startswith('lstm'):
                new_hyp_memories = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                if options['decoder'].startswith('lstm'):
                    new_hyp_memories.append(copy.copy(next_memory[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            if options['decoder'].startswith('lstm'):
                hyp_memories = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    if len(new_hyp_samples[idx]) >= minlen:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    if options['decoder'].startswith('lstm'):
                        hyp_memories.append(new_hyp_memories[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            if options['decoder'].startswith('lstm'):
                next_memory = numpy.array(hyp_memories)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])


    return sample, sample_score

def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    iterator.start()
    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=50, n_words_src=options['n_words_src'], n_words=options['n_words'])
        
        if x is None:
            continue

        pprobs = f_log_probs(x,x_mask,y,y_mask)
        for pp in pprobs:
            probs.append(pp)

        if verbose:
            print >> sys.stderr, '%d samples computed'%(n_done)

    return numpy.array(probs)

def clip_gradients(clip_c, grads):
    # Cliping gradients
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        #new_grads_2 = []    
        #for g in new_grads:
        #    new_grads_2.append(tensor.switch(g < (g * 0. + 1e-8),
        #                       g * 0., g))

        return new_grads
    return grads

def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          hiero=None,  # 'gru_hiero', # or None
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.,
          alpha_c=0.,
          diag_c=0.,
          lrate=0.01,
          n_words_src=100000,
          n_words=100000,
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='saved_models/model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq updates
          dataset='wmt14enfr',
          dictionary=None,  # word dictionary
          dictionary_src=None,  # word dictionary
          use_dropout=False,
          reload_=False,    # Contains the name of the file to reload or false
          correlation_coeff=0.1,
          clip_c=0.):

    model_options = copy.copy(inspect.currentframe().f_locals)
    model_options['decoder_FR'] = 'gru_cond_FR'
    #model_options['encoder_adversarial'] = 'gru_w_mlp'
    # model_options = locals().copy()

        # reload options
    if reload_:
        with open('{}.npz.pkl'.format(reload_), 'rb') as f:
            saved_options = pkl.load(f)
            model_options.update(saved_options)

    if model_options['dictionary']:
        word_dict, word_idict = load_dictionary(model_options['dictionary'])

    if model_options['dictionary_src']:
            word_dict_src, word_idict_src = load_dictionary(model_options['dictionary_src'])

    print 'Loading data'
    load_data, prepare_data = get_dataset(dataset)
    train, valid, test = load_data(batch_size=batch_size)



    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(reload_ + '.npz'):
        params = load_params(reload_ + '.npz', params)

    tparams = init_tparams(params)

    trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, cost_discriminator, cost_generator, B_tf, B_fr, D_o, D_f = build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]
    inps_gen_adversarial = [x, x_mask, y]

    f_B = theano.function(inps, [B_tf, B_fr])
    f_D = theano.function(inps, [D_o, D_f])
    # theano.printing.debugprint(cost.mean(), file=open('cost.txt', 'w'))

    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
                                opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after any regularizer
    print 'Building f_cost...',
    #f_cost = theano.function(inps, cost, profile=profile, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    #f_cost_discriminator = theano.function(inps, cost_discriminator, profile=profile, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    #f_cost_generator = theano.function(inps_gen_adversarial, cost_generator, profile=profile, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    f_cost = theano.function(inps, cost, profile=profile)
    f_cost_discriminator = theano.function(inps, cost_discriminator, profile=profile)
    f_cost_generator = theano.function(inps_gen_adversarial, cost_generator, profile=profile)

    print 'Done'

    if model_options['hiero'] is not None:
        print 'Building f_beta...',
        f_beta = theano.function([x, x_mask], opt_ret['hiero_betas'], profile=profile)
        print 'Done'

    print 'Computing gradient...',
    params_nll = init_params_nll(tparams)
    params_adversarial = init_params_adversarial(tparams)
    params_gen_adversarial = init_params_gen_adversarial(tparams)

    grads = tensor.grad(cost, wrt=itemlist(params_nll))
    grads_discriminator = tensor.grad(cost_discriminator, wrt=itemlist(params_adversarial))
    grads_generator = tensor.grad(cost_generator, wrt=itemlist(params_gen_adversarial))
    print 'Done'
    #print 'Building f_grad...',
    #f_grad = theano.function(inps, grads, profile=profile, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    #f_grad_discriminator = theano.function(inps, grads_discriminator, profile=profile, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    #f_grad_generator = theano.function(inps_gen_adversarial, grads_generator, profile=profile, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    f_grad = theano.function(inps, grads, profile=profile)
    f_grad_discriminator = theano.function(inps, grads_discriminator, profile=profile)
    f_grad_generator = theano.function(inps_gen_adversarial, grads_generator, profile=profile)

    #print 'Done'
    print('clip_c: {}'.format(clip_c))
    grads = clip_gradients(clip_c, grads)
    grads_discriminator = clip_gradients(clip_c, grads_discriminator)
    grads_generator = clip_gradients(clip_c, grads_generator)

    lr = tensor.scalar(name='lr')
    lr_discriminator = tensor.scalar(name='lr_discriminator')
    lr_generator = tensor.scalar(name='lr_generator')
    print 'Building optimizers...',
    # f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    f_update = eval(optimizer)(lr, params_nll, grads, inps, cost)
    f_update_discriminator = eval(optimizer)(lr_discriminator, params_adversarial, grads_discriminator, inps, cost_discriminator)
    f_update_generator = eval(optimizer)(lr_generator, params_gen_adversarial, grads_generator, inps_gen_adversarial, cost_generator)

    #BUILD ADVERSARIAL OPTIMIZER
    # f_update_adversarial = eval(optimizer)(lr, tparams, grads_adversarial, cost_adversarial)
    print 'Done'

    print 'Optimization'

    # FIXME: review this bit to make sure it is loading properly
    history_errs = []
    # Reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0]) / batch_size

    if reload_:
        model_name = reload_.split('/')[-1] 
        uidx = int(model_name.split('_')[1][5:])
        eidx_start = int(model_name.split('_')[0][5:])
    else:
        uidx = 0
        eidx_start = 0
    estop = False

    #####################
    # Main Training Loop
    #####################

    for eidx in xrange(eidx_start,max_epochs):
        n_samples = 0

        train.start()
        for x, y in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src, n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()
            '''
            c = f_cost(x, x_mask, y, y_mask)
            cd = f_cost_discriminator(x, x_mask, y, y_mask)
            cg = f_cost_generator(x, x_mask, y)
            print c
            print cd
                print cg

            g = f_grad(x, x_mask, y, y_mask)
            gd = f_grad_discriminator(x, x_mask, y, y_mask)
            gg = f_grad_generator(x, x_mask, y)
            print numpy.array([numpy.isnan(a).sum() for a in g]).sum() + numpy.array([numpy.isnan(a).sum() for a in gd]).sum() + numpy.array([numpy.isnan(a).sum() for a in gg]).sum()
            rint numpy.array([numpy.isinf(a).sum() for a in g]).sum() + numpy.array([numpy.isinf(a).sum() for a in gd]).sum() + numpy.array([numpy.isinf(a).sum() for a in gg]).sum()
            '''

            cost = f_update(x, x_mask, y, y_mask, lrate)
            D_orig, D_fake = f_D(x, x_mask, y, y_mask)
            discriminator_accuracy = ((D_fake < 0.5).sum() + (D_orig > 0.5).sum()) / (1.0 * (D_fake.size + D_orig.size))

            if discriminator_accuracy < 0.95:
                cost_discriminator = f_update_discriminator(x, x_mask, y, y_mask, lrate)
            if discriminator_accuracy > 0.75:
                cost_generator = f_update_generator(x, x_mask, y, lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                if numpy.isinf(cost):
                    print 'inf detected'
                else:
                    print 'NaN detected'

                print(uidx)
                break
                #return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
                print 'Cost Discriminator: {}, Cost Generator: {}'.format(cost_discriminator, cost_generator)
                print 'Accuracy discriminator: {}'.format(discriminator_accuracy)

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                # import ipdb; ipdb.set_trace()

                # if best_p is not None:
                #     params = best_p
                # else:
                params = unzip(tparams)

                saveto_list = saveto.split('/')
                saveto_list[-1] = 'epoch' + str(eidx) + '_' + 'nbUpd' + str(uidx) + '_' + saveto_list[-1]
                saveName = '/'.join(saveto_list)
                numpy.savez(saveName, history_errs=history_errs, **params)
                print model_options
                print saveName
                pkl.dump(model_options, open('%s.pkl' % saveName, 'wb'))
                print 'Done'

            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = False
                    sample, score = gen_sample(tparams, f_init, f_next, x[:, jj][:, None],
                                               model_options, trng=trng, k=1, maxlen=30,
                                               stochastic=stochastic, argmax=True)
                    print 'Source ', jj, ': ',
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in word_idict_src:
                            print word_idict_src[vv],
                        else:
                            print 'UNK',
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv],
                        else:
                            print 'UNK',
                    print
                    if model_options['hiero']:
                        betas = f_beta(x[:, jj][:, None], x_mask[:, jj][:, None])
                        print 'Validity ', jj, ': ',
                        for vv, bb in zip(y[:, jj], betas[:, 0]):
                            if vv == 0:
                                break
                            print bb,
                        print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv],
                        else:
                            print 'UNK',
                    print

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0
                # for _, tindex in kf:
                #     x, mask = prepare_data(train[0][train_index])
                #     train_err += (f_pred(x, mask) == train[1][tindex]).sum()
                # train_err = 1. - numpy.float32(train_err) / train[0].shape[0]

                # train_err = pred_error(f_pred, prepare_data, train, kf)
                if valid is not None:
                    valid_err = pred_probs(f_log_probs, prepare_data, model_options, valid).mean()
                if test is not None:
                    test_err = pred_probs(f_log_probs, prepare_data, model_options, test).mean()

                history_errs.append([valid_err, test_err])

                if uidx == 0 or valid_err <= numpy.array(history_errs)[:, 0].min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience, 0].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

                print 'Seen %d samples' % n_samples

        # print 'Epoch ', eidx, 'Update ', uidx, 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        # print 'Seen %d samples'%n_samples

        if estop:
            break

    if best_p is not None: 
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    test_err = 0
    #train_err = pred_error(f_pred, prepare_data, train, kf)
    if valid is not None:
        valid_err = pred_probs(f_log_probs, prepare_data, model_options, valid).mean()
    if test is not None:
        test_err = pred_probs(f_log_probs, prepare_data, model_options, test).mean()

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    if best_p is not None:
        params = copy.copy(best_p)
    else:
        params = unzip(tparams)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err,
                valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                **params)

    return train_err, valid_err, test_err



if __name__ == '__main__':
    train(dim_word=100,
          dim=500,
          encoder='gru',
          decoder='gru_cond',
          hiero=None,
          patience=10,
          max_epochs=100,
          dispFreq=10,
          decay_c=0.,
          alpha_c=0.,
          diag_c=0.,
          lrate=0.01,
          n_words_src=100000,
          n_words=100000,
          maxlen=75,
          optimizer='adadelta',
          batch_size=16,
          valid_batch_size=16,
          saveto='./saved_models/fr-en/pretrained_adversarial_simple/model.npz',
          validFreq=1000,
          saveFreq=1000,
          sampleFreq=100,
          dataset='stan',
          dictionary='../data/vocab_and_data_small_europarl_v7_enfr/vocab.en.pkl',
          dictionary_src='../data/vocab_and_data_small_europarl_v7_enfr/vocab.fr.pkl',
          use_dropout=False,
          reload_='./saved_models/fr-en/pretrained_adversarial_simple/epoch8_nbUpd151000_model',
          correlation_coeff=0.1,
          clip_c=1.)
