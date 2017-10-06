import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape, cat_entropy
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import gym

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class FcPolicy(object):

    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = conv_to_fc(X)
            h2 = fc(h1, 'fc1', nh=512, init_scale=np.sqrt(2))
            h3 = fc(h2, 'fc2', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc3', nh=128, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class RandomPolicy(object):

    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = conv_to_fc(X)
            h2 = fc(h1, 'fc1', nh=512, init_scale=np.sqrt(2))
            h3 = fc(h2, 'fc2', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc3', nh=128, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            # a, v = sess.run([a0, v0], {X:ob})
            a = np.random.randint(0, nact, nbatch)
            v = np.zeros([nbatch])
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return np.zeros([nbatch])

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class NoOpPolicy(object):

    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = conv_to_fc(X)
            h2 = fc(h1, 'fc1', nh=512, init_scale=np.sqrt(2))
            h3 = fc(h2, 'fc2', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc3', nh=128, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            # a, v = sess.run([a0, v0], {X:ob})
            a = np.zeros([nbatch], dtype='int32')
            v = np.zeros([nbatch])
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return np.zeros([nbatch])

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class ErsPolicy(object):

    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        import math
        nbases = int(math.sqrt(nact))
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = conv_to_fc(X)
            h2 = fc(h1, 'fc1', nh=512, init_scale=np.sqrt(2))
            h3 = fc(h2, 'fc2', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc3', nh=128, init_scale=np.sqrt(2))
            pi_src = tf.expand_dims(fc(h4, 'pi_src', nbases, act=lambda x:x), axis=-1) # shape = (nbatch, nbases, 1)
            pi_dst = tf.expand_dims(fc(h4, 'pi_dst', nbases, act=lambda x:x), axis=-1) # shape = (nbatch, nbases, 1)
            pi = tf.reshape(tf.matmul(pi_dst, pi_src, transpose_b=True), shape=[nbatch, nact]) # shape = (nbatch, nbases*nbases)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            #print(a)
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class ErsPolicy2(object):

    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        import math
        nbases = int(math.sqrt(nact))
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        A = tf.placeholder(tf.int32, [nbatch]) # given actions.
        with tf.variable_scope("model", reuse=reuse):
            h1 = conv_to_fc(X)
            h2 = fc(h1, 'fc1', nh=512, init_scale=np.sqrt(2))
            h3 = fc(h2, 'fc2', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc3', nh=128, init_scale=np.sqrt(2))
            pi_src = fc(h4, 'pi_src', nbases, act=lambda x:x) # shape = (nbatch, nbases)
            a_src_sampled = lambda: tf.one_hot(indices=sample(pi_src), depth=nbases) # shape = (nbatch, nbases)
            a_src_given = lambda: tf.one_hot(tf.mod(A, nbases * tf.ones([nbatch], tf.int32)), depth=nbases) # shape = (nbatch, nbases)
            a_src = tf.case([(tf.equal(A[0], tf.constant(-1, tf.int32)), a_src_sampled)],
                            default=a_src_given)
            h4_concat_a_src = tf.concat([h4, a_src], axis=-1) # shape = (nbatch, h4+nbases)
            pi_dst_given_src = fc(h4_concat_a_src, 'pi_dst_given_src', nbases, act=lambda x:x) # shape = (nbatch, nbases)
            a_dst = tf.one_hot(indices=sample(pi_dst_given_src), depth=nbases) # shape = (nbatch, nbases)
            pi = tf.reshape(tf.matmul(tf.expand_dims(pi_dst_given_src, axis=-1), tf.expand_dims(pi_src, axis=-1), transpose_b=True), shape=[nbatch, nact])
            a = tf.reshape(tf.matmul(tf.expand_dims(a_dst, axis=-1), tf.expand_dims(a_src, axis=-1), transpose_b=True), shape=[nbatch, nact])
            vf = fc(h4, 'v', 1, act=lambda x:x)
            entropy = tf.reduce_mean(cat_entropy(pi_src)) + tf.reduce_mean(cat_entropy(pi_dst_given_src))

        v0 = vf[:, 0]
        a0 = tf.argmax(a, 1)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob, A: -1*np.ones([self.nbatch], 'int32')})
            #print(a)
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.nbatch = nbatch
        self.nbases = nbases
        self.X = X
        self.A = A
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.entropy = entropy

class ErsPolicy3(object):

    def __init__(self, sess, ob_space, ob_dtype, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        import math
        nbases = int(math.sqrt(nact))
        X = tf.placeholder(ob_dtype, ob_shape) #obs
        A = tf.placeholder(tf.int32, [nbatch]) # given actions.
        with tf.variable_scope("model", reuse=reuse):
            h1 = conv_to_fc(X)
            h2 = fc(h1, 'fc1', nh=512, init_scale=np.sqrt(2))
            h3 = fc(h2, 'fc2', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc3', nh=128, init_scale=np.sqrt(2))
            pi_dst = fc(h4, 'pi_dst', nbases, act=lambda x:x) # shape = (nbatch, nbases)
            a_dst_sampled = lambda: tf.one_hot(indices=sample(pi_dst), depth=nbases) # shape = (nbatch, nbases)
            a_dst_given = lambda: tf.one_hot(tf.floordiv(A, nbases * tf.ones([nbatch], tf.int32)), depth=nbases) # shape = (nbatch, nbases)
            a_dst = tf.case([(tf.equal(A[0], tf.constant(-1, tf.int32)), a_dst_sampled)],
                            default=a_dst_given)
            h4_concat_a_dst = tf.concat([h4, a_dst], axis=-1) # shape = (nbatch, h4+nbases)
            pi_src_given_dst = fc(h4_concat_a_dst, 'pi_src_given_dst', nbases, act=lambda x:x) # shape = (nbatch, nbases)
            a_src = tf.one_hot(indices=sample(pi_src_given_dst), depth=nbases) # shape = (nbatch, nbases)
            pi = tf.reshape(tf.matmul(tf.expand_dims(pi_dst, axis=-1), tf.expand_dims(pi_src_given_dst, axis=-1), transpose_b=True), shape=[nbatch, nact])
            a = tf.reshape(tf.matmul(tf.expand_dims(a_dst, axis=-1), tf.expand_dims(a_src, axis=-1), transpose_b=True), shape=[nbatch, nact])
            vf = fc(h4, 'v', 1, act=lambda x:x)
            entropy = tf.reduce_mean(cat_entropy(pi_dst)) + tf.reduce_mean(cat_entropy(pi_src_given_dst))

        v0 = vf[:, 0]
        a0 = tf.argmax(a, 1)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob, A: -1*np.ones([self.nbatch], 'int32')})
            #print(a)
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.nbatch = nbatch
        self.nbases = nbases
        self.X = X
        self.A = A
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.entropy = entropy

