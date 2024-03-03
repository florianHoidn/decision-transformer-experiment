import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

from utils import get_optimizer
import os

import time


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment(
    exp_prefix,
    variant,
):
    torch.manual_seed(variant["seed"])
    os.makedirs(variant["outdir"], exist_ok=True)
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)

    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    group_name = f"{exp_prefix}-{env_name}-{dataset}"
    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"
    
    dataset_path = f"data/{env_name}-{dataset}-v2.pkl"
    if env_name == "hopper":
        env = gym.make("Hopper-v3")
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
    elif env_name == "halfcheetah":
        env = gym.make("HalfCheetah-v3")
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.0
    elif env_name == "walker2d":
        env = gym.make("Walker2d-v3")
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.0
    elif env_name == "reacher2d":
        from decision_transformer.envs.reacher_2d import Reacher2dEnv

        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.0
    else:
        if "MineRL" in env_name:
            import minerl
        env = gym.make(env_name)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = UnpackActionDictEnv(env)

        # TODO do we want this?
        from spinup.env_wrappers.dynamic_skip_env import DynamicSkipEnv
        env = DynamicSkipEnv(env)

        max_ep_len = 100000
        env_targets = []
        scale = 1.0
        if dataset is None or len(dataset) == 0:
            dataset_path = f"data/{env_name}.pkl"
        else:
            dataset_path = f"data/{env_name}-{dataset}.pkl"

    if model_type == "bc":
        env_targets = env_targets[
            :1
        ]  # since BC ignores target, no need for different evaluations

    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = {k:v.shape for k,v in env.observation_space.spaces.items()}
        multi_modal = True
    else:
        state_dim = env.observation_space.shape[0]
        multi_modal = False
    act_dim = env.action_space.shape[0]

    # load dataset
    subtrajectory_lenght = 15
    max_num_trajectories = 100
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
    else:
        # Let's try to optimize online.
        trajectories = []
        traj_lens, max_subtrajectory_returns = [], []
        for _ in range(1000):
            with torch.no_grad():
                ret, length, new_online_traj = evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model=None,
                    max_ep_len=2000,
                    scale=None,
                    target_return=None,
                    mode=None,
                    state_mean=None,
                    state_std=None,
                    device=device,
                    random_sampling=True
                )
            # Instead of maximizing total returns, let's just look for trajectories with nice sub-trajectories. 
            discounted_returns = discount_cumsum(new_online_traj["rewards"], gamma=0.8)
            max_subtrajectory_return = np.max(discounted_returns)
            if len(trajectories) < max_num_trajectories:
                trajectories.append(new_online_traj)
                #max_subtrajectory_returns.append(ret)
                max_subtrajectory_returns.append(max_subtrajectory_return)
                traj_lens.append(length)
            else:
                min_return_idx = np.argmin(max_subtrajectory_returns)
                min_return = max_subtrajectory_returns[min_return_idx]
                #if ret > min_return:
                if max_subtrajectory_return > min_return:
                    print(f"Adding new online trajectory with sub return {max_subtrajectory_return}")
                    trajectories[min_return_idx] = new_online_traj
                    #max_subtrajectory_returns[min_return_idx] = ret
                    max_subtrajectory_returns[min_return_idx] = max_subtrajectory_return
                    traj_lens[min_return_idx] = length


    # save all path information into separate lists
    mode = variant.get("mode", "normal")
    states, traj_lens, returns = {key:[] for key in state_dim} if multi_modal else [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        if multi_modal:
            path["observations"] = {"pov":path["observations"]["pov"]} # TODO just for testing.
            path["next_observations"] = {"pov":path["next_observations"]["pov"]} # TODO just for testing.
            
            for key in states:
                states[key].append(path["observations"][key])
        else:
            states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization

    if multi_modal:
        states = {key:np.concatenate(val, axis=0) for key, val in states.items()}

        state_mean = {key:np.mean(val, axis=0) for key, val in states.items()}
        # TODO this kinda blows up my memory without a limit
        state_std = {key:np.std(val[:500], axis=0) + 1e-6 for key, val in states.items()}
    else:
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    max_training_ret = np.max(returns)
    print("=" * 50)
    print(f"Starting new experiment: {env_name} {dataset}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {max_training_ret:.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    pct_traj = variant.get("pct_traj", 1.0)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    K = variant["K"]
    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]
    def get_batch(trajectories, returns, max_subtrajectory_returns, subtrajectory_lenght, traj_lens, max_num_trajectories, sorted_inds, state_mean, state_std, num_trajectories, p_sample, batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = {key:[] for key in state_dim} if multi_modal else [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            if multi_modal:
                for key in state_dim:
                    s[key].append(traj["observations"][key][si : si + max_len].reshape(1, -1, *state_dim[key]))
            else:
                s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            #timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1)) 
            timesteps.append(np.arange(si, si + a[-1].shape[1]).reshape(1, -1)) # TODO should be the same, right?
            
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff

            if "discount_cumsum" not in traj:
                traj["discount_cumsum"] = discount_cumsum(traj["rewards"], gamma=1.0)
            rtg.append(
                traj["discount_cumsum"][si:][:a[-1].shape[1] + 1].reshape(1, -1, 1)
            )

            #if rtg[-1].shape[1] <= s[-1].shape[1]:
            if rtg[-1].shape[1] <= a[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            #tlen = s[-1].shape[1]
            tlen = a[-1].shape[1]
            
            if multi_modal:
                for key in state_dim:
                    s[key][-1] = np.concatenate(
                        [np.zeros((1, max_len - tlen, *state_dim[key])), s[key][-1]], axis=1
                    )
                    s[key][-1] = (s[key][-1] - state_mean[key]) / state_std[key]
            else:
                s[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
                )
                s[-1] = (s[-1] - state_mean) / state_std

            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )
        if multi_modal:
            for key in state_dim:
                s[key] = torch.from_numpy(np.concatenate(s[key], axis=0)).to(
                    dtype=torch.float32, device=device
                )
        else:
            s = torch.from_numpy(np.concatenate(s, axis=0)).to(
                dtype=torch.float32, device=device
            )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):

            collected_eval_episodes = []
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == "dt":
                        ret, length, new_online_traj = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=2000,#max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length, new_online_traj = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                collected_eval_episodes.append(new_online_traj)
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
            }, collected_eval_episodes

        return fn

    if model_type == "dt":
        model = DecisionTransformer(
            args=variant,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"], # TODO that probably needs to be adjusted for proper multi modal inputs.
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=0.1,
        )
        if variant["load_checkpoint"]:
            state_dict = torch.load(variant["load_checkpoint"])
            model.load_state_dict(state_dict)
            print(f"Loaded from {variant['load_checkpoint']}")
    elif model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant["warmup_steps"]
    optimizer = get_optimizer(args=variant, model=model)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if model_type == "dt":
        trainer = SequenceTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            extra_batch_args={"trajectories":trajectories, "returns":returns, "max_subtrajectory_returns":max_subtrajectory_returns, "subtrajectory_lenght":subtrajectory_lenght, "traj_lens":traj_lens, "max_num_trajectories":max_num_trajectories, "sorted_inds":sorted_inds, "state_mean":state_mean, "state_std":state_std, "num_trajectories":num_trajectories, "p_sample":p_sample},
            variant=variant,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            eval_fn_template=eval_episodes,
            discount_cumsum=discount_cumsum,
        )
    elif model_type == "bc":
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            extra_batch_args={"trajectories":trajectories, "returns":returns, "max_subtrajectory_returns":max_subtrajectory_returns, "subtrajectory_lenght":subtrajectory_lenght, "traj_lens":traj_lens, "max_num_trajectories":max_num_trajectories, "sorted_inds":sorted_inds, "state_mean":state_mean, "state_std":state_std, "num_trajectories":num_trajectories, "p_sample":p_sample},
            variant=variant,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="decision-transformer",
            config=variant,
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        if log_to_wandb:
            wandb.log(outputs)

# TODO I'll just quickly drop this here.
class UnpackActionDictEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

         # TODO remove
        self.observation_space = gym.spaces.Dict({"pov":self.env.observation_space.spaces["pov"]})

        self.action_space_key, self.action_space = list(env.action_space.spaces.items())[0]

    def step(self, action):
        obs, total_rew, done, info = self.env.step({self.action_space_key:action})
        # TODO remove
        return {"pov":obs["pov"]}, total_rew, done, info

    def reset(self): # TODO remove
        obs = self.env.reset()
        return {"pov":obs["pov"]}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument(
        "--dataset", type=str, default="medium"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="action_noise" #"normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--model_type", type=str, default="dt"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--lm_learning_rate", "-lmlr", type=float, default=None)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--max_iters", type=int, default=400)
    parser.add_argument("--num_steps_per_iter", type=int, default=100)#2500)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained_lm", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--gpt_kmeans", type=int, default=None)
    parser.add_argument("--extend_positions", action="store_true", default=False)
    parser.add_argument("--gpt_kmeans_const", type=float, default=None)
    parser.add_argument("--kmeans_cache", type=str, default=None)

    parser.add_argument("--share_input_output_proj", action="store_true", default=False)
    parser.add_argument("--kmeans_mean", action="store_true", default=False)
    args = parser.parse_args()

    experiment("gym-experiment", variant=vars(args))
