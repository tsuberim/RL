import gym
from rl import (play, episodes, ObservationEncoder, WithCuriousity, RandomAgent, PolicyGradAgent,
                GymAdapter, AtariImageEncoder, WithHabits, TensorboardStatsWriter, ConsoleStatsWriter, WithValueEstimator, WithMemeory, PongEncoder, DenseBlock)
from gen import take, last, do
from mxnet.gluon import Trainer
from mxnet.ndarray import array
from mxnet import autograd, init
from functools import partial


def compose(*fs):
    def g(x):
        for f in fs:
            x = f(x)
        return x
    return g


def rescale(arr):
    mn = arr.min()
    mx = arr.max()
    return (arr - mn) / (mx - mn)


def main():
    tensorboard_writer = TensorboardStatsWriter(save_path='save/logs')
    console_writer = ConsoleStatsWriter(save_path='save/console')

    game = gym.make('Acrobot-v1')
    game._max_episode_steps = 2.5e3

    agent = GymAdapter(PolicyGradAgent(
        observation_space=game.observation_space,
        action_space=game.action_space,
        reward_range=game.reward_range,
        entropy_weight=5e-3,
        discount=0.995
    ))
    agent.initialize(init.Xavier())

    with agent, tensorboard_writer, console_writer:
        rollout = play(game, agent, render=True, blur_weight=0.99)
        eps = episodes(rollout)
        trainer = Trainer(agent.collect_params(), 'adam',
                          dict(learning_rate=5e-3))
        for obs, acts, rews, infos in eps:
            with autograd.record():
                loss, stats = agent.loss(obs, acts, rews, infos)
            loss.backward()
            trainer.step(len(obs))
            console_writer.write(stats)
            tensorboard_writer.write(stats)


if __name__ == '__main__':
    main()
