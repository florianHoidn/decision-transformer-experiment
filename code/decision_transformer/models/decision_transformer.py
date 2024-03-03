import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


from kmeans_pytorch import kmeans

import os


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        args,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        if args["pretrained_lm"]:
            print("Loading from pretrained")
            config = transformers.GPT2Config.from_pretrained(args["pretrained_lm"])
            config.attn_pdrop = 0.1  # args["dropout"]
            config.resid_pdrop = args["dropout"]
            self.transformer = GPT2Model.from_pretrained(
                args["pretrained_lm"],
                config=config,
            )
            if args["gpt_kmeans"] is not None:
                if args["kmeans_cache"] is not None and not os.path.exists(
                    args["kmeans_cache"]
                ):
                    cluster_ids_x, self.cluster_centers = kmeans(
                        X=self.transformer.wte.weight.data,
                        num_clusters=args["gpt_kmeans"],
                        distance="cosine",
                        device=args.get("device", "cuda"),
                    )
                    if args["kmeans_cache"] is not None:
                        torch.save(self.cluster_centers, args["kmeans_cache"])
                else:
                    self.cluster_centers = torch.load(args["kmeans_cache"])
                self.cluster_centers = self.cluster_centers.to(
                    args.get("device", "cuda")
                )
                # self.cluster_centers.requires_grad = True
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd

        else:
            config = transformers.GPT2Config(
                # vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=hidden_size,
                **kwargs,
            )

            self.transformer = GPT2Model(config)
        if max_ep_len > config.n_positions and args["extend_positions"]:
            current_max_pos, embed_size = self.transformer.wpe.weight.shape
            new_encoder_pos_embed = self.transformer.wpe.weight.new_empty(
                max_ep_len, embed_size
            )
            # copy position embeddings over and over to initialize the new position embeddings
            orig_k = 2
            k = orig_k
            step = current_max_pos - k
            new_encoder_pos_embed[:k] = self.transformer.wpe.weight[:k]
            while k < max_ep_len - 1:
                new_encoder_pos_embed[k : (k + step)] = self.transformer.wpe.weight[
                    orig_k : min(max_ep_len - k + orig_k, current_max_pos)
                ]
                k += step
            self.transformer.wpe.weight.data = new_encoder_pos_embed
        if args["frozen"]:
            for param in self.transformer.h.parameters():
                param.requires_grad = False

        if args["extend_positions"]:
            self.embed_timestep = self.transformer.wpe
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        if self.multi_modal:
            self.permute_required = set()
            self.expand_dim_required = set()
            self.modalities = []
            self.embed_state = self.create_multi_modal_embeddings(sizes_dict=state_dim, embedding_size=hidden_size)
            self.embed_sub_states = torch.nn.Linear(len(state_dim) * hidden_size, hidden_size)
        else:
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        if args["share_input_output_proj"]:
            if self.multi_modal:
                self.predict_state = {modality:lambda x: F.linear(x, self.embed_state[modality].weight.t()) for modality in self.modalities}
            else:
                self.predict_state = lambda x: F.linear(x, self.embed_state.weight.t())
            self.predict_return = lambda x: F.linear(x, self.embed_return.weight.t())
            self.predict_action = lambda x: F.tanh(
                F.linear(x, self.embed_action.weight.t())
            )
        else:
            if self.multi_modal:
                #self.predict_state = nn.ModuleDict({modality:nn.Linear(hidden_size, np.prod(self.state_dim[modality])) for modality in self.modalities})
                self.predict_state = {modality:lambda x: x for modality in self.modalities} # TODO For this we would need a deconvolution. Do we need this?
            else:
                self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
            self.predict_return = torch.nn.Linear(hidden_size, 1)

        self.past_key_values = None
        print(self)

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        past_key_values=None,
    ):

        #batch_size, seq_length = states.shape[0], states.shape[1]
        batch_size, seq_length = returns_to_go.shape[0], returns_to_go.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        if self.multi_modal:
            sub_embeddings = []
            for modality in self.modalities:
                if modality in self.permute_required:
                    sub_in = torch.reshape(states[modality], shape=(batch_size * seq_length,) + states[modality].shape[2:]).permute(0,3,1,2)
                elif modality in self.expand_dim_required:
                    sub_in = torch.unsqueeze(torch.reshape(states[modality], shape=(batch_size * seq_length,) + states[modality].shape[2:]), dim=1)
                else:
                    sub_in = torch.reshape(states[modality], shape=(batch_size * seq_length,) + states[modality].shape[2:])
                sub_embeddings.append(torch.reshape(self.embed_state[modality](sub_in), shape=(batch_size, seq_length, self.hidden_size)))
            # TODO integrate sub embeddings into the sequence properly.
            # TODO For now, I might just integrate everything into one embedding.
            state_embeddings = self.embed_sub_states(torch.cat(sub_embeddings, dim=-1))
            #state_embeddings = sub_embeddings[0]
        else:
            state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        # state_embeddings = state_embeddings + time_embeddings
        # action_embeddings = action_embeddings + time_embeddings
        # returns_embeddings = returns_embeddings + time_embeddings

        # TODO the permutes etc. need to be adjusted for proper multiple input modalities

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        all_embs = self.embed_ln(stacked_inputs)

        stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            past_key_values=None,  # self.past_key_values,
            use_cache=True,
        )
        x = transformer_outputs["last_hidden_state"]
        self.past_key_values = transformer_outputs["past_key_values"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(
            x[:, 2]
        )  # predict next return given state and action
        # predict next state given state and action:
        if self.multi_modal:
            state_preds = {modality:self.predict_state[modality](
                x[:, 2]
            ) for modality in self.modalities}
        else:
            state_preds = self.predict_state(
                x[:, 2]
            )  
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds, all_embs

    def get_action(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        past_key_values=None,
        **kwargs
    ):
        # we don't care about the past rewards in this model

        if self.multi_modal:
            states = {modality:states[modality].reshape(1, -1, *self.state_dim[modality]) for modality in self.modalities}
        else:
            states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            if self.multi_modal:
                states = {modality:states[modality][:, -self.max_length :] for modality in self.modalities}
            else:
                states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length

            attention_mask = torch.cat(
                [
                    #torch.zeros(self.max_length - states.shape[1]),
                    #torch.ones(states.shape[1]),
                    torch.zeros(self.max_length - returns_to_go.shape[1]),
                    torch.ones(returns_to_go.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                #dtype=torch.long, device=states.device
                dtype=torch.long, device=returns_to_go.device
            ).reshape(1, -1)

            if self.multi_modal:
                states = {modality:torch.cat(
                    [
                        torch.zeros(
                            (
                                states[modality].shape[0],
                                self.max_length - states[modality].shape[1],
                                *self.state_dim[modality],
                            ),
                            device=states[modality].device,
                        ),
                        states[modality],
                    ],
                    dim=1,
                ).to(dtype=torch.float32) for modality in self.modalities}
            else:
                states = torch.cat(
                    [
                        torch.zeros(
                            (
                                states.shape[0],
                                self.max_length - states.shape[1],
                                self.state_dim,
                            ),
                            device=states.device,
                        ),
                        states,
                    ],
                    dim=1,
                ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, __ = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )

        return action_preds[0, -1]

    def create_multi_modal_embeddings(self, sizes_dict, embedding_size, conv_sizes=[(32, 8, 4, 0, 1), (64, 4, 2, 0, 1), (128, 2, 2, 0, 1)]):
        input_dict = {}
        for modality, size in sizes_dict.items():
            self.modalities.append(modality)
            if type(size) == list or type(size) == tuple:
                size = np.array(size)

            if np.ndim(size) == 0:
                input_dict[modality] = nn.Linear(size, embedding_size)
            elif size.shape[0] in [2, 3]:
                # let's treat 2d and 3d input tensors as images.
                if size.shape[0] == 3:
                    if size[2] in [1,3]:
                        in_channels = size[2]
                        self.permute_required.add(modality)
                    else:
                        # Otherwise, the color channel hopefully already is in the first position, as pytorch requires it.
                        in_channels = size[0]
                else:
                    in_channels = 1
                    self.expand_dim_required.add(modality)
                specialized_sub_net = []
                conv_out_size = np.array(size[:-1]) if modality in self.permute_required else np.array(size[1:]) # See section "Shape" at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html 
                for nbr_filters, kernel_size, stride, padding, dilation in conv_sizes:
                    specialized_sub_net.append(nn.Conv2d(in_channels=in_channels, out_channels=nbr_filters, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
                    specialized_sub_net.append(nn.ReLU())
                    conv_out_size = np.floor((conv_out_size + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1)
                    in_channels = nbr_filters        
                specialized_sub_net.append(nn.Flatten())
                specialized_sub_net.append(nn.Linear(int(np.prod(conv_out_size) * in_channels), embedding_size))
                input_dict[modality] = nn.Sequential(*specialized_sub_net)
            elif size.shape[0] == 1:
                input_dict[modality] = nn.Linear(size[0], embedding_size)
            else:
                print("MultiModalNet doesn't yet know how to handle input of size " + str(size) + ". Do the input images have the shape (H,W,C)?")
        return nn.ModuleDict(input_dict)
