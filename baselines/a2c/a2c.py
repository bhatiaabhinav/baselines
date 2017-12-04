import os.path as osp
import gym
import time
import joblib
import json
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.policies import CnnPolicy, ErsPolicy2, ErsPolicy3
from baselines.a2c.utils import cat_entropy, mse

class Model(object):

    def __init__(self, policy, ob_space, ob_dtype, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear', ecschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        EC = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ob_dtype, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ob_dtype, ac_space, nenvs, nsteps, nstack, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        if policy in [ErsPolicy2, ErsPolicy3]:
            entropy = train_model.entropy
        else:
            entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * EC + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        #if max_grad_norm is not None:
        #    grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        #trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        trainer = tf.train.AdamOptimizer(learning_rate=LR)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        ec = Scheduler(v=ent_coef, nvalues=total_timesteps, schedule=ecschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            advs -= np.mean(advs)
            advs /= (np.std(advs) + 0.01)
            for step in range(len(obs)):
                cur_lr = lr.value()
                cur_ec = ec.value()
            if policy in [ErsPolicy2, ErsPolicy3]:
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, EC:cur_ec, train_model.A:actions}
            else:
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, EC:cur_ec}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, ob_dtype='uint8', nsteps=5, nstack=4, gamma=0.99, _lambda=1.0, render=False):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=ob_dtype)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self._lambda = _lambda
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.render = render

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_info = [],[],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            #self.env.render(mode='human') if self.render else self.env.render(close=True)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards)
            mb_info.extend(_)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, values, value) in enumerate(zip(mb_rewards, mb_dones, mb_values, last_values)):
            values_appended = np.asarray(values.tolist() + [value if dones[-1]==0 else 0])
            dones_appended = np.asarray(dones.tolist() + [0])
            one_step_advantages = rewards + (1 - dones_appended[:-1]) * self.gamma * values_appended[1:] - values_appended[:-1]
            generalized_advantages = discount_with_dones(one_step_advantages, dones, self.gamma * self._lambda)
            # generalized_advantages becomes same as (nstep discounted returns - values) when _lambda=1 (as in standard A2C/A3C implementation)
            # advantage clipping:
            generalized_advantages = np.clip(generalized_advantages, -1, 1)
            rewards = values + np.asarray(generalized_advantages)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_info

def learn(policy, env, seed, ob_dtype='uint8', nsteps=5, nstack=4, total_timesteps=int(80e6), frameskip=1, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=2.5e-4, lrschedule='linear', ecschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, _lambda=1.0, log_interval=100, saved_model_path=None, render=False, no_training=False):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK

    if logger.get_dir():
        with open(osp.join(logger.get_dir(), 'params.json'), 'w') as f:
            f.write(json.dumps({'policy':str(policy), 'env_id':env.id, 'nenvs':nenvs, 'seed':seed, 'ac_space':str(ac_space), 'ob_space':str(ob_space), 'ob_type':ob_dtype, 'nsteps':nsteps, 'nstack':nstack, 'total_timesteps':total_timesteps, 'frameskip':frameskip, 'vf_coef':vf_coef, 'ent_coef':ent_coef,
                                'max_grad_norm':max_grad_norm, 'lr':lr, 'lrschedule':lrschedule, 'ecschedule':ecschedule, 'epsilon':epsilon, 'alpha':alpha, 'gamma':gamma, 'lambda':_lambda, 'log_interval':log_interval, 'saved_model_path':saved_model_path, 'render':render, 'no_training':no_training, 'abstime': time.time()}))

    model = Model(policy=policy, ob_space=ob_space, ob_dtype=ob_dtype, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, ecschedule=ecschedule)
    if saved_model_path:
        try:
            model.load(saved_model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Model could not be loaded:\n{0}".format(e))
    
    runner = Runner(env, model, ob_dtype=ob_dtype, nsteps=nsteps, nstack=nstack, gamma=gamma, _lambda=_lambda, render=render)

    nbatch = nenvs*nsteps
    tstart = time.time()
    logger.record_tabular('t_start', tstart)
    logger.dump_tabular()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, info = runner.run()
        if not no_training:
            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        else:
            policy_loss, value_loss, policy_entropy = 0.0, 0.0, 0.0
        nseconds = time.time()-tstart
        sps = int((update*nbatch)/nseconds) #timesteps per second
        fps = frameskip * sps
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular('t', nseconds)
            logger.record_tabular("nupdates", 0 if no_training else update)
            logger.record_tabular("total_timesteps", update*nbatch) # for backward compatibility
            logger.record_tabular("fps", fps)
            logger.record_tabular("ntimesteps", update*nbatch)
            logger.record_tabular("nframes", update*nbatch*frameskip)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("average_value", float(np.average(values)))
            if 'base0' in info[0]:
                for i in range(20):
                    base = 'base{0}'.format(i)
                    if base in info[0]:
                        alloc = np.average([inf[base] for inf in info])
                        logger.record_tabular("av_" + base, alloc)
            if 'ambs_relocating' in info[0]:
                reloc = np.average([inf['ambs_relocating'] for inf in info])
                logger.record_tabular("av_ambs_relocating", reloc)
            action0 = (float(len(actions) - np.count_nonzero(actions)) * 100) / len(actions)
            logger.record_tabular("action0_percentage", action0)
            logger.dump_tabular()
            if logger.get_dir(): model.save(osp.join(logger.get_dir(), "model"))
    if logger.get_dir(): model.save(osp.join(logger.get_dir(), "model"))
    env.close()

if __name__ == '__main__':
    main()
