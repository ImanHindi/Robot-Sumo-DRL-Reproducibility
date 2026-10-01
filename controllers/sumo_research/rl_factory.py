from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _algo_class(name: str):
    key = name.strip().lower()
    if key == "a2c":
        from stable_baselines3 import A2C
        return A2C
    if key == "ppo":
        from stable_baselines3 import PPO
        return PPO
    if key == "dqn":
        from stable_baselines3 import DQN
        return DQN
    if key in {"recurrentppo", "rppo", "recurrent_ppo"}:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    raise ValueError(f"Unsupported algorithm: {name}")


def normalized_name(name: str) -> str:
    key = name.strip().lower()
    return {
        "a2c": "A2C",
        "ppo": "PPO",
        "dqn": "DQN",
        "recurrentppo": "RecurrentPPO",
        "rppo": "RecurrentPPO",
        "recurrent_ppo": "RecurrentPPO",
    }[key]


def create_model(algo_name: str, env, cfg: Dict[str, Any], seed: int, tensorboard_log: str | None):
    algo = normalized_name(algo_name)
    cls = _algo_class(algo)
    params = dict(cfg.get("algorithms", {}).get(algo, {}))
    net_arch = params.pop("policy_net", [128, 128])

    common = dict(env=env, verbose=1, seed=seed, tensorboard_log=tensorboard_log)
    if algo == "A2C":
        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))
        return cls("MlpPolicy", policy_kwargs=policy_kwargs, **params, **common)
    if algo == "PPO":
        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))
        return cls("MlpPolicy", policy_kwargs=policy_kwargs, **params, **common)
    if algo == "DQN":
        policy_kwargs = dict(net_arch=net_arch)
        return cls("MlpPolicy", policy_kwargs=policy_kwargs, **params, **common)
    if algo == "RecurrentPPO":
        lstm_hidden_size = int(params.pop("lstm_hidden_size", 128))
        n_lstm_layers = int(params.pop("n_lstm_layers", 1))
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=True,
            enable_critic_lstm=False,
        )
        return cls("MlpLstmPolicy", policy_kwargs=policy_kwargs, **params, **common)
    raise AssertionError(algo)


def load_model(algo_name: str, model_path: str | Path, env=None, tensorboard_log: str | None = None):
    algo = normalized_name(algo_name)
    cls = _algo_class(algo)
    kwargs = {}
    if env is not None:
        kwargs["env"] = env
    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log
    return cls.load(str(model_path), **kwargs)


def predict_action(model, algo_name: str, obs, deterministic: bool, state=None, episode_start=None):
    algo = normalized_name(algo_name)
    if algo == "RecurrentPPO":
        action, state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return action, state
    action, _ = model.predict(obs, deterministic=deterministic)
    return action, state
