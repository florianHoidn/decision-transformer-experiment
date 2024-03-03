import numpy as np
import torch
import tqdm
import time


class Trainer:
    def __init__(
        self,
        args,
        model,
        optimizer,
        batch_size,
        get_batch,
        extra_batch_args,
        variant,
        loss_fn,
        scheduler=None,
        eval_fns=None,
        eval_fn_template=None,
        discount_cumsum=None,
        eval_only=False,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scaler = torch.cuda.amp.GradScaler()
        self.get_batch = get_batch
        self.extra_batch_args = extra_batch_args
        self.variant = variant
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.step = 0
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.eval_fn_template = eval_fn_template
        self.discount_cumsum = discount_cumsum
        self.diagnostics = dict()
        self.eval_only = eval_only

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        if not self.eval_only:
            self.model.train()
            for _ in tqdm.tqdm(range(num_steps), desc="Training"):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

                logs["time/training"] = time.time() - train_start
                logs["training/train_loss_mean"] = np.mean(train_losses)
                logs["training/train_loss_std"] = np.std(train_losses)

        eval_start = time.time()

        self.model.eval()
        all_new_episodes = []
        if self.eval_fn_template is None:
            dynamic_env_targets = []
        else:
            max_known_ret = np.max(self.extra_batch_args["returns"])
            dynamic_env_targets = [self.eval_fn_template(0.5 * max_known_ret), self.eval_fn_template(2 * max_known_ret)]
        for eval_fn in tqdm.tqdm(self.eval_fns + dynamic_env_targets, desc="Evaluating"):
            outputs, new_episodes = eval_fn(self.model)
            all_new_episodes += new_episodes
            for k, v in outputs.items():
                logs[f"evaluation/{k}"] = v

        if not self.eval_only:
            self.update_trajectories(all_new_episodes)
            logs["time/total"] = time.time() - self.start_time
        logs["time/evaluation"] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        if not self.eval_only:
            if self.args.get("outdir"):
                torch.save(
                    self.model.state_dict(),
                    #f"{self.args['outdir']}/model_{iter_num}.pt",
                    f"{self.args['outdir']}/model.pt",
                )

        return logs

    def train_step(self):
        self.optimizer.zero_grad()
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(
            self.batch_size
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        if self.args["fp16"]:
            with torch.cuda.amp.autocast():

                state_preds, action_preds, reward_preds = self.model.forward(
                    states,
                    actions,
                    rewards,
                    masks=None,
                    attention_mask=attention_mask,
                    target_return=returns,
                )

                # note: currently indexing & masking is not fully correct
                loss = self.loss_fn(
                    state_preds,
                    action_preds,
                    reward_preds,
                    state_target[:, 1:],
                    action_target,
                    reward_target[:, 1:],
                )
        else:

            state_preds, action_preds, reward_preds = self.model.forward(
                states,
                actions,
                rewards,
                masks=None,
                attention_mask=attention_mask,
                target_return=returns,
            )

            # note: currently indexing & masking is not fully correct
            loss = self.loss_fn(
                state_preds,
                action_preds,
                reward_preds,
                state_target[:, 1:],
                action_target,
                reward_target[:, 1:],
            )

        if self.args["fp16"]:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.detach().cpu().item()

    def update_trajectories(self, new_trajectories):
        trajectories = self.extra_batch_args["trajectories"]
        max_num_traj = self.extra_batch_args["max_num_trajectories"]
        #returns = self.extra_batch_args["returns"]
        max_subtrajectory_returns = self.extra_batch_args["max_subtrajectory_returns"] 
        subtrajectory_lenght = self.extra_batch_args["subtrajectory_lenght"]
        if len(trajectories) < max_num_traj:
            trajectories += new_trajectories
        else:
            for new_traj in new_trajectories:
                #ret = new_traj["rewards"].sum()
                # Instead of maximizing total returns, let's just look for trajectories with nice sub-trajectories. 
                discounted_returns = self.discount_cumsum(new_traj["rewards"], gamma=0.8)
                max_subtrajectory_return = np.max(discounted_returns)
                #min_return_idx = np.argmin(returns)
                #min_return = returns[min_return_idx]
                min_return_idx = np.argmin(max_subtrajectory_returns)
                min_return = max_subtrajectory_returns[min_return_idx]
                #if ret > min_return:
                if max_subtrajectory_return > min_return:
                    print(f"Adding new online trajectory with sub return {max_subtrajectory_return}")
                    trajectories[min_return_idx] = new_traj
                    #returns[min_return_idx] = ret
                    max_subtrajectory_returns[min_return_idx] = max_subtrajectory_return

        # Essentially copied from experiment.py
        mode = self.variant.get("mode", "normal")
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            if mode == "delayed":  # delayed: all rewards moved to end of trajectory
                path["rewards"][-1] = path["rewards"].sum()
                path["rewards"][:-1] = 0.0
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum()) # TODO That's kinda redundant. 
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Added {len(new_trajectories)} new trajectories to training data.")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print("=" * 50)

        pct_traj = self.variant.get("pct_traj", 1.0)

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

        self.extra_batch_args["trajectories"] = trajectories
        self.extra_batch_args["returns"] = returns
        self.extra_batch_args["max_subtrajectory_returns"] = max_subtrajectory_returns
        self.extra_batch_args["traj_lens"] = traj_lens
        self.extra_batch_args["sorted_inds"] = sorted_inds
        # TODO pass mean and std into the evaluation, too.
        self.extra_batch_args["state_mean"] = state_mean
        self.extra_batch_args["state_std"] = state_std
        self.extra_batch_args["num_trajectories"] = num_trajectories
        self.extra_batch_args["p_sample"] = p_sample
