import gym
from rl import RandomAgent, PolicyGradientAgent, play, episodes, returns, AtariImageEncoder, IntrinsicCuriosityModule, ConsoleStatsWriter, TensorboardStatsWriter
from gen import take, last
from random import uniform
from mxnet.ndarray import array, one_hot
from mxnet import autograd
from mxnet.gluon import Block, Trainer, loss


def main():
    game = gym.make('MontezumaRevenge-v0')
    with PolicyGradientAgent(game.action_space, save_path='./save/agent', observation_encoder=AtariImageEncoder(16)) as agent,\
            TensorboardStatsWriter(save_path='./save/logs') as writer:
        rollout = play(game, agent, render=True)
        eps = episodes(rollout)
        for obs, acts, rews, infos in eps:
            obs, acts, rews = array(obs), array(acts), array(rews)
            with autograd.record():
                err, stats = agent.loss(obs, acts, rews)
            err.backward()
            agent.trainer.step(1)
            writer.write(dict(
                **stats,
                total_return=returns(rews)[0].asscalar(),
                total_steps=len(rews)
            ))

    game.close()


if __name__ == '__main__':
    main()
