import os
import json
import numpy as np
from mxnet import autograd, init
from mxnet.ndarray import *
from mxnet.gluon import Block, Trainer, loss
from mxnet.gluon.nn import Sequential, Dense, Flatten, BatchNorm, Conv2D, MaxPool2D, Activation
from tqdm import tqdm


class Persistent:
    def __init__(self, save_path=None):
        self.save_path = save_path

    def _get_save_path(self, save_path=None, label=None):
        save_path = save_path or self.save_path
        if label:
            save_path = '%s-%s' % (save_path, label)
        return save_path

    def _save(self, path):
        raise NotImplementedError()

    def _load(self, path):
        raise NotImplementedError()

    def save(self, save_path=None, label=None):
        save_path = self._get_save_path(save_path, label)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._save(save_path)

    def load(self, save_path=None, label=None):
        save_path = self._get_save_path(save_path, label)
        self._load(save_path)

    def __enter__(self):
        if self.save_path:
            try:
                self.load()
            except Exception as e:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_path:
            self.save()


class Agent:
    def step(self, ob, rew):
        raise NotImplementedError()

    def reset(self):
        pass


class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def step(self, ob, rew):
        return self.action_space.sample()


class NNAgent(Agent, Block, Persistent):
    def __init__(self, save_path=None, observation_encoder=None, **kwargs):
        Block.__init__(self, **kwargs)
        Persistent.__init__(self, save_path)
        self.save_path = save_path
        self.observation_encoder = observation_encoder or (lambda x: x)
        self.state = None

    def step(self, ob, rew):
        ob = self.observation_encoder(array(ob).expand_dims(axis=0))
        action, self.state = self._step(self.state, ob, rew)
        return action

    def reset(self):
        self.state = None

    def _step(self, state, ob, rew):
        raise NotImplementedError()

    def _save(self, path):
        self.save_parameters(path)

    def _load(self, path):
        self.load_parameters(path)


class PolicyGradientAgent(NNAgent):
    def __init__(
        self,
        action_space,
        lr=1e-2,
        entropy_weight=1e-3,
        discount=0.99,
        n_hidden=16,
        n_layers=2,
        **kwargs
    ):
        NNAgent.__init__(self, **kwargs)
        self.entropy_weight = entropy_weight
        self.discount = discount

        self.icm = IntrinsicCuriosityModule(action_space)
        self.policy = Sequential()
        self.policy.add(
            Flatten(),
            *[Dense(n_hidden, activation='relu') for _ in range(n_layers)],
            Dense(action_space.n)
        )
        self.policy.initialize(init.Xavier())

        self.trainer = Trainer(
            self.collect_params(),
            'adam',
            dict(learning_rate=lr)
        )

    def forward(self, ob):
        return softmax(self.policy(ob))

    def _step(self, prev_ob, ob, rew):
        prev_ob = prev_ob if prev_ob is not None else ob
        act_dist = self(ob)
        action = sample_multinomial(act_dist)
        return int(action.asscalar()), ob

    def loss(self, obs, acts, rews):
        encs = self.observation_encoder(obs)

        curiosity_rews = concat(
            self.icm(encs[:-1], acts[:-1], encs[1:]),
            array([0]),
            dim=0
        )
        rets = returns(rews + curiosity_rews, self.discount)
        act_dists = self(encs)
        entropy = -sum(act_dists*log(act_dists), axis=1)
        icm_err, icm_stats = self.icm.loss(encs, acts)
        policy_grad_err = -log(act_dists.pick(acts))*rets
        err = policy_grad_err \
            - self.entropy_weight*entropy \
            + icm_err

        return err.mean(), dict(
            **icm_stats,
            policy_grad_err=policy_grad_err.mean().asscalar(),
            entropy=entropy.mean().asscalar(),
            total_err=err.mean().asscalar(),
        )


def play(game, agent=None, render=False):
    agent = agent or RandomAgent(game.action_space)
    while True:
        agent.reset()
        ob, rew, done = game.reset(), 0, False
        while not done:
            if render:
                game.render()
            act = agent.step(ob, rew)
            new_ob, rew, done, info = game.step(act)
            yield ob, act, rew, done, info
            ob = new_ob


def episodes(rollout):
    obs, rews, acts, infos = [], [], [], []
    for ob, act, rew, done, info in rollout:
        obs.append(ob)
        acts.append(act)
        rews.append(rew)
        infos.append(info)
        if done:
            yield obs, acts, rews, infos
            obs, acts, rews, infos = [], [], [], []


def returns(rewards, discount=1):
    rewards = array(rewards).asnumpy()
    discounts = discount**np.arange(len(rewards))
    discounted_rewards = rewards*discounts
    return array(np.cumsum(discounted_rewards[::-1])[::-1] / discounts)


class StatsWriter(Persistent):
    def __init__(self, **kwargs):
        Persistent.__init__(self, **kwargs)
        self.idx = 0

    def _save(self, path):
        with open(path, 'w') as f:
            json.dump(self.idx, f)

    def _load(self, path):
        with open(path, 'r') as f:
            self.idx = json.load(f)

    def _write(self, idx, key, value):
        pass

    def write(self, stats):
        for k, v in stats.items():
            self._write(self.idx, k, v)
        self.idx += 1


class ConsoleStatsWriter(StatsWriter):
    def __init__(self, **kwargs):
        StatsWriter.__init__(self, **kwargs)

    def _write(self, idx, key, value):
        print('{0: >4}|{1: >10} ={2:8.4f}'.format(idx, key, value))


class TensorboardStatsWriter(StatsWriter):
    def __init__(self, save_path=None, **kwargs):
        if not save_path:
            raise ValueError('save_path not specified')
        from mxboard import SummaryWriter
        logdir = save_path
        save_path = os.path.join(logdir, 'save')
        os.makedirs(logdir, exist_ok=True)
        self.sw = SummaryWriter(logdir=logdir)
        StatsWriter.__init__(self, save_path=save_path, **kwargs)

    def _write(self, idx, key, value):
        self.sw.add_scalar(key, value, idx)


class VisdomStatsWriter(StatsWriter):
    def __init__(self, **kwargs):
        StatsWriter.__init__(**kwargs)
        from visdom import Visdom
        self.vis = Visdom()

    def _write(self, idx, key, value):
        self.vis.line(
            np.array([value]),
            np.array([idx]),
            win=key,
            update='append',
            opts=dict(title=key)
        )


class AtariImageEncoder(Block):
    def __init__(self, n_dims=128, **kwargs):
        Block.__init__(self, **kwargs)
        if n_dims < 16:
            raise ValueError(
                '`n_dims` must be at least 16 (given: %d)' % n_dims)

        self.encoder = Sequential()
        self.encoder.add(
            Conv2D(int(n_dims/16), 6, (4, 3)),
            BatchNorm(),
            MaxPool2D(),
            Activation('relu'),
            Conv2D(int(n_dims/8), 3),
            Activation('relu'),
            Conv2D(int(n_dims/2), 3),
            BatchNorm(),
            MaxPool2D(),
            Activation('relu'),
            Conv2D(int(n_dims), 3),
            MaxPool2D(),
            Activation('relu'),
            Flatten()
        )
        self.encoder.initialize(init.Xavier())

    def forward(self, img):
        ob = (img.transpose((0, 3, 1, 2)) - 128) / 255
        enc = self.encoder(ob[:, :, ::2, ::2])
        return enc


class IntrinsicCuriosityModule(Block):
    def __init__(self, action_space, lr=1e-3, n_dims=16, n_hidden=16, n_layers=8):
        Block.__init__(self)
        self.action_space = action_space

        self.encoder = Sequential()
        self.forward_model = Sequential()
        self.inverse_model = Sequential()

        for seq in (self.encoder, self.forward_model, self.inverse_model):
            seq.add(
                Flatten(),
                *[Dense(n_hidden, activation='relu') for _ in range(n_layers)],
                Dense(n_dims)
            )
            seq.initialize(init.Xavier())

        self.pred_enc_loss, self.pred_act_loss = loss.L1Loss(
            100), loss.SoftmaxCrossEntropyLoss()
        self.trainer = Trainer(
            self.collect_params(),
            'adam',
            dict(learning_rate=lr)
        )

    def forward(self, prev_ob, act, ob):
        prev_enc, enc = self.encoder(prev_ob), self.encoder(ob)
        pred_enc = self.forward_model(
            concat(prev_enc, one_hot(act, self.action_space.n)))
        return self.pred_enc_loss(pred_enc, enc)

    def loss(self, obs, acts, act_err_weight=0.8):
        encs = self.encoder(obs)
        pred_encs = self.forward_model(
            concat(encs[:-1], one_hot(acts[:-1], self.action_space.n)))
        pred_acts = self.inverse_model(concat(encs[:-1], encs[1:]))
        pred_enc_err = self.pred_enc_loss(pred_encs, encs[1:])
        pred_act_err = self.pred_act_loss(pred_acts, acts[:-1])
        err = (1-act_err_weight)*pred_enc_err + act_err_weight*pred_act_err
        return err.mean(), dict(
            pred_enc_err=pred_enc_err.mean().asscalar(),
            pred_act_err=pred_act_err.mean().asscalar(),
            icm_err=err.mean().asscalar(),
        )
