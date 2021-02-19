from tensortrade.oms.instruments import Instrument, BTC, USD
from tensortrade.env.default.actions import BSH

from tensortrade.env.generic import Renderer

import matplotlib.pyplot as plt
import numpy as np

import ray
import numpy as np
import pandas as pd

import tensortrade.env.default as default

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

from tensortrade.data.cdd import CryptoDataDownload

from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

import ray
from ray.rllib.utils.filter import MeanStdFilter

import torch


def create_env(config, train="train"):
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
    if False:
        data.close = data.close / 20 + range(len(data))
        print("genenrating fake increase")
    if train == "train":
        data = data[0:int(len(data) / 2)]  # training
        print("using first half for training")
    elif train == "eval":
        data = data[int(len(data) / 2):]  # validation
        print("using second half for eval")
    else:
        print("using all data")

    pclose = Stream.source(list(data.close), dtype="float").rename("USD-BTC")
    pmin = Stream.source(list(data.low), dtype="float").rename("USD-BTClow")
    pmax = Stream.source(list(data.high), dtype="float").rename("USD-BTChigh")

    pmin = Stream.source(list(data.low), dtype="float").rename("USD-BTClow")
    pmax = Stream.source(list(data.high), dtype="float").rename("USD-BTChigh")

    pmin3 = pmin.rolling(window=3).min()
    pmin10 = pmin.rolling(window=10).min()
    pmin20 = pmin.rolling(window=20).min()
    pmax3 = pmax.rolling(window=3).max()
    pmax10 = pmax.rolling(window=10).max()
    pmax20 = pmax.rolling(window=20).max()

    eo = ExchangeOptions(commission=0.002)  #
    coinbase = Exchange("coinbase", service=execute_order, options=eo)(
        pclose
    )

    cash = Wallet(coinbase, 100000 * USD)
    asset = Wallet(coinbase, 0 * BTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    feed = DataFeed([

        (pclose.log() - pmin3.log()).fillna(0).rename("relmin3"),
        (pclose.log() - pmin10.log()).fillna(0).rename("relmin10"),
        (pclose.log() - pmin20.log()).fillna(0).rename("relmin20"),
        (pclose.log() - pmax3.log()).fillna(0).rename("relmax3"),
        (pclose.log() - pmax10.log()).fillna(0).rename("relmax10"),
        (pclose.log() - pmax20.log()).fillna(0).rename("relmax20"),

    ])

    action_scheme = BSH(cash=cash, asset=asset)

    renderer_feed = DataFeed([
        Stream.source(list(data.close), dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")  # only works for BSH
    ])

    environment = default.create(

        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme="simple",
        renderer_feed=renderer_feed,
        renderer=PositionChangeChart(),
        window_size=config["window_size"],
        min_periods=20,
        max_allowed_loss=0.6
    )
    return environment


# Get checkpoint
window_size = 30
pname = "PPO_TradingEnv_8c527_00000_0_2021-02-16_13-21-51"
# c:\work\git\tensortrade\examples\myexample\Experiments\PPO\PPO_TradingEnv_8c527_00000_0_2021-02-16_13-21-51\checkpoint_2000\
checkpoint_path = "c:/work/klemen/rlagent/Experiments/PPO/" + pname + "/checkpoint_2000/checkpoint-2000"

ray.init(local_mode=True)

# Restore agent
agent = ppo.PPOTrainer(
    #env="TradingEnv",
    config={
        #"env": "TradingEnv",  #this gives:   File "C:\Python\Python38\lib\site-packages\gym\envs\registration.py", line 118, in spec
                                        #raise error.Error('Attempted to look up malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id.encode('utf-8'), env_id_re.pattern))
                                        #gym.error.Error: Attempted to look up malformed environment ID: b'TradingEnv'. (Currently all IDs must be of the form ^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$.)
        
        "env": "TradingEnv-v1",
        "env_config": {
            "window_size": window_size
        },
        "model": {
            # Share layers for value function. If you set this to True, it's
            # important to tune vf_loss_coeff.
            # "vf_share_layers": True,
            "vf_share_layers": False,
            "fcnet_hiddens": [32, 16, 16],

            "use_lstm": True,
            # Max seq len for training the LSTM, defaults to 20.
            "max_seq_len": 20,
            # Size of the LSTM cell.
            # "lstm_cell_size": 256,
            "lstm_cell_size": 32,
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            "lstm_use_prev_action": False,  # TODO: play with this
            # Whether to feed r_{t-1} to LSTM.
            "lstm_use_prev_reward": False,
            # Experimental (only works with `_use_trajectory_view_api`=True):
            # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
            "_time_major": False,
        },
        "log_level": "DEBUG",
        "framework": "torch",
        "ignore_worker_failures": True,
        "num_workers": 3,
        'num_gpus': 1,
        "clip_rewards": False,
        "gamma": 0,
        "observation_filter": "MeanStdFilter",
        # "observation_filter": MyStdFilter,  # demean=False, destd=True rewrd = -3.04045e+06 after 1000
        # "observation_filter": "NoFilter",     # => reward -1.12313e+08 after 1000 iterations
        "lambda": 0.72,
        # "vf_loss_coeff": 0.5,
        "vf_loss_coeff": 1.0,
        # "entropy_coeff": 0.01
        "entropy_coeff": 0.1,
        "lr": 5e-5,

        # 'rollout_fragment_length': 300,
    }
)
agent.restore(checkpoint_path)

# Visualization
# Instantiate the environment
env = create_env({
    "window_size": window_size
}, "train")

episode_reward = 0
done = False
obs = env.reset()

j = 0

state = agent.get_policy().model.get_initial_state()

while not done:
    action, state, logits = agent.compute_action(obs, state=state)
    obs, reward, done, info = env.step(action)
    print(j, action, reward, done, info)
    episode_reward += reward
    j += 1

env.render()

###
# Instantiate the environment
env = create_env({
    "window_size": window_size
}, "eval")

episode_reward = 0
done = False
obs = env.reset()

j = 0

state = agent.get_policy().model.get_initial_state()

while not done:
    action, state, logits = agent.compute_action(obs, state=state)
    obs, reward, done, info = env.step(action)
    print(j, action, reward, done, info)
    episode_reward += reward
    j += 1

env.render()
