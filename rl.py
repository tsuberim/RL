import os
import json
import numpy as np
from mxnet import autograd, init
from mxnet.ndarray import *
from mxnet.gluon import Block, Trainer, loss
from mxnet.gluon.nn import Sequential, Dense, Flatten, BatchNorm, Conv2D, MaxPool2D, Activation
from mxnet.gluon.rnn import GRU
from tqdm import tqdm
from gym.envs.classic_control import rendering


epsilon = 1e-6


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


class PersistentBlock(Block, Persistent):
    def __init__(self, save_path=None, **kwargs):
        Persistent.__init__(self, save_path=save_path)
        Block.__init__(self, **kwargs)

    def _save(self, path):
        self.save_parameters(path)

    def _load(self, path):
        self.load_parameters(path)


class AtariImageEncoder(PersistentBlock):
    def __init__(self, n_dims=128, **kwargs):
        PersistentBlock.__init__(self, **kwargs)
        if n_dims < 16:
            raise ValueError(
                '`n_dims` must be at least 16 (given: %d)' % n_dims)

        self.encoder = Sequential()
        self.encoder.add(
            BatchNorm(),
            Conv2D(int(n_dims/16), 6, (4, 3)),
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
            Conv2D(int(n_dims), 3),
            MaxPool2D(),
            Activation('relu'),
            Flatten()
        )

    def forward(self, img):
        transposed = img.transpose((0, 3, 1, 2))
        downscaled = transposed[:, :, ::2, ::2]
        normalized = (downscaled - 128) / 255
        enc = self.encoder(normalized)
        return enc


class PongEncoder(PersistentBlock):
    def __init__(self, **kwargs):
        PersistentBlock.__init__(self, **kwargs)
        self.prev = None

    def forward(self, img):
        cropped = img[:, 35:195]
        downscaled = cropped[:, ::2, ::2, 0]
        removed_bg = (1 - (downscaled == 144))*downscaled
        removed_bg2 = (1 - (removed_bg == 109))*removed_bg
        removed_bg3 = removed_bg2 != 0
        enc = removed_bg3
        if img.shape[0] > 1:
            prev = concat(zeros_like(enc[0]).expand_dims(
                axis=0), enc[:-1], dim=0)
            out = enc - prev
            self.prev = None
        else:
            if self.prev is None:
                self.prev = zeros_like(enc)
            out = enc - self.prev
            self.prev = enc
        return out.reshape((0, -1))


class DenseBlock(PersistentBlock):
    def __init__(
        self,
        n_dims=16,
        n_hidden_units=16,
        n_hidden_layers=2,
        activation='relu',
        transform=(lambda x: x),
        ** kwargs
    ):
        PersistentBlock.__init__(self, **kwargs)
        self.transform = transform
        self.seq = Sequential()
        self.seq.add(
            Flatten(),
            *[Dense(n_hidden_units, activation=activation)
              for _ in range(n_hidden_layers)],
            Dense(n_dims)
        )

    def forward(self, x):
        return self.transform(self.seq(x))


class Agent(PersistentBlock):
    def __init__(self, observation_space, action_space, reward_range, discount=0.99, **kwargs):
        PersistentBlock.__init__(self, **kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range
        self.discount = float(discount)

    def step(self, ob, rew):
        raise NotImplementedError()

    def reset(self):
        pass

    def loss(self, obs, acts, rews, infos):
        raise NotImplementedError()


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, ob, rew):
        return self.action_space.sample(), {}


class AgentWrapper(Agent):
    def __init__(self, agent, **kwargs):
        super().__init__(
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            reward_range=agent.reward_range,
            save_path=agent.save_path,
            discount=agent.discount,
            ** kwargs
        )
        self.agent = agent

    def step(self, ob, rew):
        return self.agent.step(ob, rew)

    def reset(self):
        return self.agent.reset()

    def loss(self, obs, acts, rews, infos):
        return self.agent.loss(obs, acts, rews, infos)


class GymAdapter(AgentWrapper):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)

    def step(self, ob, rew):
        ob = array(ob).expand_dims(axis=0)
        action, stats = super().step(ob, rew)
        return int(action.asscalar()), stats

    def loss(self, obs, acts, rews, infos, **kwargs):
        obs, acts, rews = array(obs), array(acts), array(rews)
        loss, stats = super().loss(obs, acts, rews, infos, **kwargs)
        return loss, dict(
            **stats,
            loss=loss,
            total_return=returns(rews)[0],
            total_steps=len(obs)
        )


class ObservationEncoder(AgentWrapper):
    def __init__(self, agent, encoder=(lambda x: x), **kwargs):
        super().__init__(agent, **kwargs)
        self.encoder = encoder

    def step(self, ob, rew):
        enc = self.encoder(ob)
        action, stats = super().step(enc, rew)
        return action, stats

    def loss(self, obs, acts, rews, infos, **kwargs):
        obs, acts, rews = array(obs), array(acts), array(rews)
        encs = self.encoder(obs)
        loss, stats = super().loss(encs, acts, rews, infos, **kwargs)
        return loss, stats


class WithCuriousity(AgentWrapper):
    def __init__(
        self,
        agent,
        encoder=DenseBlock(),
        forward_model=DenseBlock(),
        inverse_model=DenseBlock(),
        action_pred_weight=0.8,
        curiosity_rews_weight=0.8,
        ** kwargs
    ):
        super().__init__(agent, **kwargs)
        self.action_pred_weight = float(action_pred_weight)
        self.curiosity_rews_weight = float(curiosity_rews_weight)

        self.encoder = encoder
        self.forward_model = forward_model
        self.inverse_model = Sequential()
        self.inverse_model.add(
            inverse_model,
            DenseBlock(
                n_dims=self.action_space.n,
                n_hidden_layers=1,
                transform=softmax
            )
        )
        self.softmax_loss = loss.SoftmaxCrossEntropyLoss()

    def loss(self, obs, acts, rews, infos, **kwargs):
        encs = self.encoder(obs)
        one_hot_acts = one_hot(acts, self.action_space.n)
        enc_preds = self.forward_model(concat(encs[:-1], one_hot_acts[:-1]))
        action_preds = self.inverse_model(concat(encs[:-1], encs[1:]))
        action_pred_loss = self.softmax_loss(action_preds, acts[:-1])
        surprise = cosine_distance(enc_preds, encs[1:])
        extra_rews = concat(surprise, array([0]), dim=0)
        total_rews = \
            (1 - self.curiosity_rews_weight)*rews + \
            self.curiosity_rews_weight*extra_rews
        loss, stats = super().loss(obs, acts, total_rews, infos, **kwargs)
        curiosity_loss = \
            (1-self.action_pred_weight)*surprise \
            + self.action_pred_weight*action_pred_loss
        return loss + curiosity_loss.sum(), dict(
            **stats,
            surprise=surprise,
            action_prediction_loss=action_pred_loss
        )


class PolicyGradAgent(Agent):
    def __init__(self, policy=DenseBlock(), entropy_weight=1e-2, **kwargs):
        Agent.__init__(self, **kwargs)
        self.entropy_weight = entropy_weight

        self.policy = Sequential()
        self.policy.add(
            policy,
            DenseBlock(
                n_dims=self.action_space.n,
                n_hidden_layers=1,
                transform=softmax
            )
        )

    def forward(self, ob):
        return self.policy(ob)

    def step(self, ob, rew):
        act_dist = self(ob)
        action = sample_multinomial(act_dist)
        return action, dict(
            entropy=entropy(act_dist)
        )

    def loss(self, obs, acts, rews, infos):
        rets = normalize(returns(rews, self.discount))
        act_dists = self(obs)
        entropy_loss = entropy(act_dists)
        policy_grad_loss = -log(act_dists.pick(acts) + epsilon)*rets
        loss = policy_grad_loss - self.entropy_weight*entropy_loss
        return loss, dict(
            entropy=entropy_loss,
            policy_grad_loss=policy_grad_loss,
        )


class WithValueEstimator(AgentWrapper):
    def __init__(self, agent, predictor=DenseBlock(), **kwargs):
        super().__init__(agent, **kwargs)

        self.predictor = Sequential()
        self.predictor.add(
            predictor,
            DenseBlock(
                n_dims=1,
                n_hidden_layers=1,
            )
        )

    def loss(self, obs, acts, rews, infos, **kwargs):
        return_pred_loss = (returns(rews, self.discount) -
                            self.predictor(obs))**2
        predicted_rest_return = self.predictor(obs[-1].expand_dims(axis=0))[0]
        new_rews = concat(rews[:-1], rews[-1] + predicted_rest_return, dim=0)
        loss, stats = super().loss(obs, acts, new_rews, infos, **kwargs)
        return loss + return_pred_loss, dict(
            **stats,
            return_pred_loss=return_pred_loss
        )


class WithMemeory(AgentWrapper):
    def __init__(self, agent, memory=GRU(16), **kwargs):
        super().__init__(agent, **kwargs)

        self.memory = memory

    def reset(self):
        super().reset()
        self.state = self.memory.begin_state(batch_size=1)

    def step(self, ob, rew):
        mem, self.state = self.memory(ob.expand_dims(axis=1), self.state)
        action, stats = super().step(mem, rew)
        return action, stats

    def loss(self, obs, acts, rews, infos, **kwargs):

        mems, states = self.memory(obs.expand_dims(
            axis=1), self.memory.begin_state(1))
        loss, stats = super().loss(mems.squeeze(), acts, rews, infos, **kwargs)
        return loss, stats


class WithHabits(AgentWrapper):
    def __init__(self, agent, predictor=DenseBlock(), entropy_weight=1e-2, **kwargs):
        super().__init__(agent, **kwargs)
        self.entropy_weight = entropy_weight

        self.predictor = Sequential()
        self.predictor.add(
            predictor,
            DenseBlock(
                n_dims=self.action_space.n,
                n_hidden_layers=1,
                transform=softmax
            )
        )

    def reset(self):
        self.cum_rew = 0
        self.habit_steps = 0
        return super().reset()

    def step(self, ob, rew):
        act_dist = self.predictor(ob)
        ent = entropy(act_dist)
        habit_prob = 1 - ent
        if random.uniform(0, 1) < habit_prob:
            self.cum_rew += rew
            self.habit_steps += 1
            return sample_multinomial(act_dist), dict(
                habit=True,
                habit_prob=habit_prob,
                cum_rew=self.cum_rew,
                habit_steps=self.habit_steps
            )
        else:
            action, stats = super().step(ob, self.cum_rew)
            self.cum_rew = 0
            self.habit_steps = 0
            return action, dict(
                **stats,
                habit=False,
                habit_prob=habit_prob.asscalar(),
                cum_rew=self.cum_rew,
                habit_steps=self.habit_steps
            )

    def loss(self, obs, acts, rews, infos):
        pred_act_dists = self.predictor(obs)
        prediction_loss = -log(pred_act_dists.pick(acts) + epsilon)
        entropy_loss = entropy(pred_act_dists)
        loss, stats = super().loss(obs, acts, rews, infos)
        return loss + prediction_loss - self.entropy_weight*entropy_loss, dict(
            **stats,
            habit_prediction_loss=prediction_loss,
            habit_prediction_entropy=entropy_loss,
            habit_steps=len([info for info in infos if info['habit']])
        )


def play(game, agent=None, render=False, blur_weight=0, render_trail=True):
    agent = agent or RandomAgent(
        observation_space=game.observation_space,
        action_space=game.action_space,
        reward_range=game.reward_range,
    )
    if render:
        viewer = rendering.SimpleImageViewer()
        img = None
    while True:
        agent.reset()
        ob, rew, done = game.reset(), 0, False
        while not done:
            if render:
                frame = game.render(mode='rgb_array') / 255
                if img is None:
                    img = frame
                if render_trail:
                    diff = (img - frame)**2
                    frame = (1-diff)*img + diff*frame
                else:
                    frame = frame
                img = blur_weight*img + (1-blur_weight)*frame
                viewer.imshow((255*img).astype(np.uint8))
            act, stats = agent.step(ob, rew)
            new_ob, rew, done, info = game.step(act)
            yield ob, act, rew, done, dict(**info, **stats)
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


def entropy(dists):
    return -sum(dists*log(dists + epsilon), axis=1) / log(array([dists.shape[1]]))


def normalize(arr):
    arr_mean = arr - mean(arr)
    arr_var = mean(arr_mean**2)
    return arr_mean / sqrt(arr_var + epsilon)


def cosine_distance(a, b):
    a1 = expand_dims(a, axis=1)
    b1 = expand_dims(b, axis=2)
    d = batch_dot(a1, b1)[:, 0, 0]
    a_norm = sqrt(sum((a*a), axis=1))
    b_norm = sqrt(sum((b*b), axis=1))
    dist = 1.0 - d / (a_norm * b_norm)
    return dist


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
            if isinstance(v, NDArray):
                self._write(self.idx, k, v.mean().asscalar())
            else:
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
        StatsWriter.__init__(self, **kwargs)
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
