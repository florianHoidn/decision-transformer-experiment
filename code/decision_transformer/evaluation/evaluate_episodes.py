import numpy as np
import torch


def evaluate_episode(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    device="cuda",
    target_return=None,
    mode="normal",
    state_mean=0.0,
    state_std=1.0,
):

    model.eval()
    model.to(device=device)

    multi_modal = type(state_mean) is dict
    if multi_modal:
        state_mean = {key:torch.from_numpy(val).to(device=device) for key, val in state_mean.items()}
        state_std = {key:torch.from_numpy(val).to(device=device) for key, val in state_std.items()}
    else:
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    if multi_modal:
        states = {key:(
            torch.from_numpy(val)
            .reshape(1, *state_dim[key])
            .to(device=device, dtype=torch.float32)
        ) for key, val in state.items()}
    else:
        states = (
            torch.from_numpy(state)
            .reshape(1, state_dim)
            .to(device=device, dtype=torch.float32)
        )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        if multi_modal:
            action = model.get_action(
                {(states[key].to(dtype=torch.float32) - state_mean[key]) / state_std[key] for key in state_mean.keys()},
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return=target_return,
            )
        else:
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return=target_return,
            )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        if multi_modal:
            cur_state = {key:torch.from_numpy(val).to(device=device).reshape(1, state_dim) for key, val in state.items()}
            states = {key:torch.cat([states[key], val], dim=0) for key, val in cur_state.items()}
        else:
            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

best_act_noise_sigma = None
best_act_noise_rew = 0
def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    random_sampling=False
):
    global best_act_noise_sigma
    global best_act_noise_rew
    multi_modal = type(state_mean) is dict
    state = env.reset()
    if not random_sampling:
        model.eval()
        model.to(device=device)
        if multi_modal:
            state_mean = {key:torch.from_numpy(val).to(device=device) for key, val in state_mean.items()}
            state_std = {key:torch.from_numpy(val).to(device=device) for key, val in state_std.items()}
        else:
            state_mean = torch.from_numpy(state_mean).to(device=device)
            state_std = torch.from_numpy(state_std).to(device=device)

        if mode == "noise":
            if multi_modal:
                state = {key:val + np.random.normal(0, 0.1, size=val.shape) for key, val in state.items()}
            else:
                state = state + np.random.normal(0, 0.1, size=state.shape)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        if multi_modal:
            states = {key:(
                torch.from_numpy(val)
                .reshape(1, *state_dim[key])
                .to(device=device, dtype=torch.float32)
            ) for key, val in state.items()}
        else:
            states = (
                torch.from_numpy(state)
                .reshape(1, state_dim)
                .to(device=device, dtype=torch.float32)
            )
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = target_return
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
            1, 1
        )
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    # I'll collect the online episodes so that we can feed them back into the training iterations.
    offline_training_keys = ["observations", "next_observations", "actions", "rewards", "terminals"]
    prev_state = state
    
    if multi_modal:
        # TODO
        new_online_trajectory = {k:[] for k in offline_training_keys}
    else:
        new_online_trajectory = {k:[] for k in offline_training_keys}

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        if random_sampling:
            action = env.action_space.sample()
        else:
            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            if multi_modal:
                action = model.get_action(
                    {key:(states[key].to(dtype=torch.float32) - state_mean[key]) / state_std[key] for key in state_mean.keys()},
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            else:
                action = model.get_action(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            actions[-1] = action
            action = action.detach().cpu().numpy()

        if mode == "action_noise":
            if best_act_noise_sigma is None:
                best_act_noise_sigma = np.ones_like(action) * 0.3
            act_noise_sigma = np.random.normal(np.abs(best_act_noise_sigma), 0.01)
            action = action + np.random.normal(0, np.abs(act_noise_sigma), size=action.shape)

        state, reward, done, _ = env.step(action)

        if mode == "action_noise" and (reward > best_act_noise_rew or (reward > best_act_noise_rew * 0.95 and np.linalg.norm(act_noise_sigma) < np.linalg.norm(best_act_noise_sigma))):
            best_act_noise_sigma = act_noise_sigma
            best_act_noise_rew = reward
            print(f"new best_act_noise: {best_act_noise_sigma}")

        if multi_modal:
            # TODO
            pass
        else:
            new_online_trajectory["observations"].append(prev_state)
            new_online_trajectory["next_observations"].append(state)
            new_online_trajectory["actions"].append(action)
            new_online_trajectory["rewards"].append(reward)
            new_online_trajectory["terminals"].append(done)
        prev_state = state

        env.render() # TODO Added some visualization. Maybe make configurable.

        if not random_sampling:
            if multi_modal:
                cur_state = {key:torch.from_numpy(val).to(device=device).reshape(1, *state_dim[key]) for key, val in state.items()}
                states = {key:torch.cat([states[key], val], dim=0) for key, val in cur_state.items()}
            else:
                cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if mode != "delayed":
                pred_return = target_return[0, -1] - (reward / scale)
            else:
                pred_return = target_return[0, -1]
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
                dim=1,
            )

        episode_return += reward
        episode_length += 1

        if done:
            break

    if not random_sampling:
        model.past_key_values = None

    return episode_return, episode_length, {k:np.array(v) for k,v in new_online_trajectory.items()}
