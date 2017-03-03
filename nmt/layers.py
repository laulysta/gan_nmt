import theano
import theano.tensor as tensor

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'ff_nb': ('param_init_fflayer_nb', 'fflayer_nb'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'gru_cond_simple': ('param_init_gru_cond_simple', 'gru_cond_simple_layer'),
          'gru_hiero': ('param_init_gru_hiero', 'gru_hiero_layer'),
          'rnn': ('param_init_rnn', 'rnn_layer'),
          'rnn_cond': ('param_init_rnn_cond', 'rnn_cond_layer'),
          'rnn_hiero': ('param_init_rnn_hiero', 'rnn_hiero_layer'),
          }

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, 
            state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
            state_before * 0.5)
    return proj


def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])


def fflayer_nb(tparams, state_below, options, prefix='ff_nb', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix, 'W')]))


def rnn_layer(tparams, state_below, options, prefix='rnn', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[0]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step(m_, xx_, h_, Ux):
        preactx = tensor.dot(h_, Ux)
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h#, r, u, preact, preactx

    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_belowx],
                                outputs_info = [tensor.alloc(0., n_samples, dim)],
                                                #None, None, None, None],
                                non_sequences=[tparams[_p(prefix, 'Ux')]], 
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)
    rval = [rval]
    return rval


def rnn_cond_layer(tparams, state_below, options, prefix='rnn', mask=None, context=None, one_step=False, init_memory=None, init_state=None, context_mask=None, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Ux')].shape[0]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix,'b_att')]
    pctx_ += tparams[_p(prefix,'b_att')]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_belowx += tparams[_p(prefix, 'bx')]
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, xx_, xc_, h_, ctx_, alpha_, pctx_,
              Wd_att, U_att, c_tt, Ux, Wcx):
        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context * alpha[:, :, None]).sum(0) # current context

        preactx = tensor.dot(h_, Ux)
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        h = tensor.tanh(preactx)

        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T  #, pstate_, preact, preactx, r, u

    if one_step:
        rval = _step(mask, state_belowx, state_belowc, init_state, None, None, 
                     pctx_, tparams[_p(prefix,'Wd_att')],
                     tparams[_p(prefix,'U_att')],
                     tparams[_p(prefix, 'c_tt')],
                     tparams[_p(prefix, 'Ux')],
                     tparams[_p(prefix, 'Wcx')] )
    else:
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_belowx, state_belowc],
                                    outputs_info = [init_state, 
                                                    tensor.alloc(0., n_samples, context.shape[2]),
                                                    tensor.alloc(0., n_samples, context.shape[0])],
                                                    #None, None, None, 
                                                    #None, None],
                                    non_sequences=[pctx_,
                                                   tparams[_p(prefix,'Wd_att')],
                                                   tparams[_p(prefix,'U_att')],
                                                   tparams[_p(prefix, 'c_tt')],
                                                   tparams[_p(prefix, 'Ux')],
                                                   tparams[_p(prefix, 'Wcx')]
                                                   ],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile)
    return rval


def rnn_hiero_layer(tparams, context, options, prefix='rnn_hiero', context_mask=None, **kwargs):

    nsteps = context.shape[0]
    if context.ndim == 3:
        n_samples = context.shape[1]
    else:
        n_samples = 1

    # mask
    if context_mask == None:
        mask = tensor.alloc(1., context.shape[0], 1)
    else:
        mask = context_mask

    dim = tparams[_p(prefix, 'Ux')].shape[0]

    # initial/previous state
    init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix,'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, h_, ctx_, alpha_, v_, pctx_,
              Wd_att, U_att, c_tt, Ux, Wx, bx, W_st, b_st):

        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pctx_ + pstate_[None,:,:] 
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context * alpha[:,:,None]).sum(0) # current context

        preactx = tensor.dot(h_, Ux)
        preactx += tensor.dot(ctx_, Wx)
        preactx += bx

        h = tensor.tanh(preactx)

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        # compute stopping probability
        ss = tensor.nnet.sigmoid(tensor.dot(h, W_st) + b_st)
        v_ = v_ * (1. - ss)[:,0][:,None]

        return h, ctx_, alpha.T, v_[:,0] #, pstate_, preact, preactx, r, u

    rval, updates = theano.scan(_step, 
                                sequences=[mask],
                                outputs_info = [init_state, 
                                                tensor.alloc(0., n_samples, context.shape[2]),
                                                tensor.alloc(0., n_samples, context.shape[0]),
                                                tensor.alloc(1., n_samples)],
                                                #None, None, None, 
                                                #None, None],
                                non_sequences=[pctx_,
                                               tparams[_p(prefix,'Wd_att')],
                                               tparams[_p(prefix,'U_att')],
                                               tparams[_p(prefix, 'c_tt')],
                                               tparams[_p(prefix, 'Ux')],
                                               tparams[_p(prefix, 'Wx')],
                                               tparams[_p(prefix, 'bx')],
                                               tparams[_p(prefix, 'W_st')],
                                               tparams[_p(prefix, 'b_st')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)

    rval[0] = rval[0] * rval[3][:,:,None]
    return rval


def gru_layer(tparams, state_below, options, prefix='gru', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h#, r, u, preact, preactx

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step, 
                                sequences=seqs,
                                outputs_info = [tensor.alloc(0., n_samples, dim)],
                                                #None, None, None, None],
                                non_sequences = [tparams[_p(prefix, 'U')], 
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


def gru_cond_simple_layer(tparams, state_below, options, prefix='gru', mask=None, context=None, one_step=False, init_memory=None, init_state=None, context_mask=None, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 2, 'Context must be 2-d: #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc')])
    pctxx_ = tensor.dot(context, tparams[_p(prefix,'Wcx')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, pctx_, pctxx_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_
        preact += pctx_
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += pctxx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]] 

    if one_step:
        rval = _step(*(seqs+[init_state, pctx_, pctxx_]+shared_vars))
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info=[init_state], 
                                    non_sequences=[pctx_,
                                                   pctxx_]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


def gru_cond_layer(tparams, state_below, options, prefix='gru', mask=None, context=None, one_step=False, init_memory=None, init_state=None, context_mask=None, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    #state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])
    #import ipdb; ipdb.set_trace()

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):

        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)   # (4)
        u1 = _slice(preact1, 1, dim)   # (5)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)  # (3)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_     # (2)

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        # pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])     # (8)
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)     # (7)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)     # (6) current context

        preact2 = tensor.dot(h1, U_nl) + b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl) + bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T #, pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step( * (seqs + [init_state, None, None, pctx_, context] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples, context.shape[2]),
                                                  tensor.alloc(0., n_samples, context.shape[0])],
                                    non_sequences=[pctx_, context] + shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


