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

print(torch.zeros(1).cuda())
print("Torch-cuda available?: " + str(torch.cuda.is_available()))


class PositionChangeChart(Renderer):

    def __init__(self, color: str = "orange"):
        self.color = "orange"

    def render(self, env, **kwargs):
        history = pd.DataFrame(env.observer.renderer_history)

        actions = list(history.action)
        p = list(history.price)

        buy = {}
        sell = {}

        for i in range(len(actions) - 1):
            a1 = actions[i]
            a2 = actions[i + 1]

            if a1 != a2:
                if a1 == 0 and a2 == 1:
                    buy[i] = p[i]
                else:
                    sell[i] = p[i]

        buy = pd.Series(buy)
        sell = pd.Series(sell)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle("Performance")

        axs[0].plot(np.arange(len(p)), p, label="price", color=self.color)
        axs[0].scatter(buy.index, buy.values, marker="^", color="green")
        axs[0].scatter(sell.index, sell.values, marker="^", color="red")
        axs[0].set_title("Trading Chart")

        performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')  # changed ISO
        performance.plot(ax=axs[1])
        axs[1].set_title("Net Worth")

        plt.show()


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


register_env("TradingEnv", create_env)

window_size = 30

ray.init(local_mode=True)

# Get checkpoint
pname = "PPO_TradingEnv_442e0_00000_0_2021-02-15_02-51-05"
# c:\work\klemen\rlagent\Experiments\PPO\PPO_TradingEnv_0e8da_00000_0_2021-02-14_18-35-40
checkpoint_path = "c:/work/klemen/rlagent/Experiments/PPO/" + pname + "/checkpoint_1080/checkpoint-1080"

analysis = tune.run(
    "PPO",
    stop={
        "episode_reward_mean": 2e15,
        "training_iteration": 2000
    },
    config={
        "env": "TradingEnv",
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
        # "num_workers": 3,  #max
        "num_workers": 3,
        "num_gpus": 1,
        "clip_rewards": False,

        "lr": 5e-5,
        "gamma": 0,

        "observation_filter": "MeanStdFilter",

        "lambda": 0.72,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.1,

    },
    # checkpoint_freq=200,  # new
    checkpoint_freq=20,  # new
    checkpoint_at_end=True,
    # restore="c:\work\klemen\rlagent\Experiments\",
    local_dir='c:/work/klemen/rlagent/Experiments',
    restore=checkpoint_path
)
