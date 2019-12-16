# ============================================================================
# Module implementing a replay memory buffer
#
# In general I try to add typing to all functions, see:
#   https://docs.python.org/3/library/typing.html
#
# Some references:
#   - https://github.com/transedward/pytorch-dqn (which were taken from the
#       Berkeley deep RL course)
#   - Google dopamine (in tf, should cross-check against this at some point
#       to make sure the replication is exact):
#       https://github.com/google/dopamine/blob/master/dopamine/replay_memory/circular_replay_buffer.py
#
# NOTE: Nov 30 2019
#   augmenting it with VQ-VAE for compression
#
# Author: Anthony G. Chen
# ============================================================================

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from vqvae_one import VQVAE


class CircularReplayBuffer(object):
    """
    Circular replay buffer for naive uniform sampling of the recent past

    With default dtypes, the DQN (Mnih 2015) full implementation with 84*84
    frame size and 1mil frames should take around 7.1 GB to store
    """

    def __init__(self, buffer_cap=50000, history=4,
                 obs_shape=(1, 84, 84), obs_dtype=torch.uint8,
                 device='cpu') -> None:
        """
        Initializing the circular replay buffer. For the atari setting, the
        internal buffer should have shapes:
            _obs_buffer : (capacity, c, 84, 84)
            _act_buffer : (capacity, 1)
            _rew_buffer : (capacity, 1)
            _done_buffer: (capacity, 1)

        :param buffer_cap: total capacity of the buffer
        :param history: number of observation to stack to make a state
        :param obs_shape: shape of observation tensor
        :param device: device to store memory on
        """

        # Initialize counter
        self._cur_idx = 0  # Current buffer index to write to
        self.size = 0  # Number of experiences stored
        self.capacity = buffer_cap  # Total buffer capacity
        self.history = history  # History length (# frames for a state)
        self._device = device  # Device (cpu/cuda) to store buffers on

        # Initialize the experience shapes and types
        self._obs_shape = obs_shape
        self._obs_dtype = obs_dtype
        self._act_dtype = torch.int32
        self._rew_dtype = torch.float32

        # Initialize the experience buffers
        obs_buffer_shape = ((self.capacity,) + self._obs_shape)
        self._obs_buffer = torch.empty(obs_buffer_shape, dtype=self._obs_dtype,
                                       device=self._device)
        self._act_buffer = torch.empty((self.capacity, 1), dtype=self._act_dtype,
                                       device=self._device)
        self._rew_buffer = torch.empty((self.capacity, 1), dtype=self._rew_dtype,
                                       device=self._device)
        self._done_buffer = torch.empty((self.capacity, 1), dtype=torch.bool,
                                        device=self._device)

    def push(self, observ: torch.tensor, action: torch.tensor,
             reward: torch.tensor, done: torch.tensor) -> None:
        """
        Pushes an experience to the buffer
        """
        # Write experiences to buffer
        # TODO: add assertions to check for types?
        self._obs_buffer[self._cur_idx] = observ
        self._act_buffer[self._cur_idx] = action
        self._rew_buffer[self._cur_idx] = reward
        self._done_buffer[self._cur_idx] = done

        self._cur_idx = (self._cur_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) \
            -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """
        Sample minibatch experience from the buffer
        :param n: batch size
        :return: batches of state, action, reward and next state
        """
        # Sample indeces, (inclusive) last indeces of the sequences
        # TODO: modify such one cannot sample index i where done[i-1]==True
        #   this is because if i is the last obs of successor state then the
        #   previous state is completely absent
        indeces = np.random.choice(self.size, size=n, replace=False)

        # Get the buffered experiences
        state_batch, next_state_batch = self.encode_states(indeces)
        act_batch = self._act_buffer[indeces]
        rew_batch = self._rew_buffer[indeces]

        return state_batch, act_batch, rew_batch, next_state_batch

    def encode_states(self, idxs: np.ndarray) -> (torch.tensor, torch.tensor):
        """
        Generate the state (stacks of observations) given the indeces of the
        last observation, zero-padding if crossing episode boundaries
        :param idxs: indeces of the last observation of the successor state
        :return: minibatches of state and successor states
        """

        # Initialize current and next states tensors (pre zero-padded)
        _state_tensor_shape = ((len(idxs), self.history * self._obs_shape[0]) + self._obs_shape[-2:])
        cur_states = torch.zeros(_state_tensor_shape,
                                 dtype=self._obs_dtype,
                                 device=self._device)
        nex_states = torch.zeros(_state_tensor_shape,
                                 dtype=self._obs_dtype,
                                 device=self._device)

        # Fill each state
        for i, buf_idx in enumerate(idxs):
            # Get the valid obs sequence of length history + 1
            seq_idxs = self._get_valid_seq((buf_idx + 1) % self.size)
            cur_seq_idxs = seq_idxs[:-1]
            nex_seq_idxs = seq_idxs[1:] if len(seq_idxs) > self.history \
                else seq_idxs

            # Fill states
            # NOTE reshaping stacks multiple channels (if present) AND frames
            #      into the 2nd dimension of the returned state tensor; being
            #      mindful that the first few indeces may be zero-padding
            if len(cur_seq_idxs) > 0:
                # number of (non zero-padded) frames and channels to stack
                non_pad_len = len(cur_seq_idxs) * self._obs_shape[0]
                cur_states[i, -non_pad_len:] = torch.reshape(
                    self._obs_buffer[cur_seq_idxs],
                    (-1, self._obs_shape[-1], self._obs_shape[-1])
                )

            if len(nex_seq_idxs) > 0:
                non_pad_len = len(nex_seq_idxs) * self._obs_shape[0]
                nex_states[i, -non_pad_len:] = torch.reshape(
                    self._obs_buffer[nex_seq_idxs],
                    (-1, self._obs_shape[-1], self._obs_shape[-1])
                )

        return cur_states, nex_states

    def _get_valid_seq(self, last_idx: int) -> List[int]:
        """
        Helper method to return a sequence of valid (observation) indeces
        (inclusive) to form (both) the previous and successor states.
        Length range from be 1 to self.history+1

        :param last_idx: last index of the sequence
        """
        assert last_idx < self.size

        # Get the allowable first index of this sequence
        first_idx = last_idx
        for j in range(1, self.history + 1):
            cur_idx = last_idx - j
            # If it goes out of bound, determine how to treat
            if cur_idx < 0:
                if self.size == self.capacity:
                    cur_idx = cur_idx % self.capacity
                else:
                    break
            # If it crosses an episode boundary
            if self._done_buffer[cur_idx]:
                break
            # If all good then keep going
            first_idx = cur_idx

        # Get the sequence of indeces
        valid_seq = []
        while first_idx != ((last_idx + 1) % self.capacity):
            valid_seq.append(first_idx)
            first_idx = (first_idx + 1) % self.capacity

        return valid_seq

    def __len__(self) -> int:
        return self.size


class VqVaeModel(object):
    """
    VQ-VAE model to store samples

    Uses the modified Sonnet VAE code
    """

    def __init__(self, in_channel=1,
                 channel=128,
                 n_res_block=2,
                 n_res_channel=32,
                 embed_dim=64,
                 n_embed=512,
                 device="cpu",
                 buffer_cap=50000,
                 history=4,
                 latent_loss_weight=0.25,
                 ):
        self.device = device
        self.embed_dim = embed_dim
        self.history = history
        self.model = VQVAE(in_channel=in_channel,
                           channel=channel,
                           n_res_block=n_res_block,
                           n_res_channel=n_res_channel,
                           embed_dim=embed_dim,
                           n_embed=n_embed
                           ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=1e-3)
        self.criterion = nn.MSELoss()
        self.latent_loss_weight = latent_loss_weight

        # VQ-VAE's own buffer
        self.buffer = CircularReplayBuffer(
            buffer_cap=buffer_cap,
            history=history,
            obs_shape=(self.embed_dim, 21, 21),  # 21 is result of convolution
            obs_dtype=torch.float32,
            device=self.device
        )

    def train(self, batch: torch.tensor):
        """
        Pushes a mini-batch of example to the VQ-VAE for training.
        Assumes batch is in the device already.

        :param batch: minibatch of training examples
        :return: average loss of minibatch
        """

        # Forward pass
        self.model.zero_grad()
        recon_batch, latent_loss = self.model(batch)

        # Compute loss and back-prop
        recon_loss = self.criterion(recon_batch, batch)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss
        loss.backward()

        self.optimizer.step()

        # mse_sum += recon_loss.item() * img.shape[0]  #TODO old code delete commented
        # mse_n += img.shape[0]

        return recon_loss.item(), latent_loss.item()

    def encode(self, batch: torch.tensor):
        """
        Encodes a mini-batch of example using a VQ-VAE
        :param batch:
        :return:
        """

        with torch.no_grad():
            quant_b, _, _ = self.model.encode(batch)
            return quant_b

    def decode(self, batch: torch.tensor):
        """
        Decode a mini-batch of examples
        :param batch:
        :return:
        """
        with torch.no_grad():
            return self.model.decode(batch)

    def push(self, observ: torch.tensor, action: torch.tensor,
             reward: torch.tensor, done: torch.tensor) -> None:
        # Encode observation
        quant_obs = self.encode(observ)
        # Push
        self.buffer.push(quant_obs, action, reward, done)

    def sample(self, n: int):
        # Sample the buffer
        # NOTE: output latent states will have shape:
        #       [batch size, history length * channel, latent h,  latent w]
        sample_out = self.buffer.sample(n)
        latent_s_batch, act_batch, rew_batch, latent_ns_batch = sample_out

        # Reshape to decode
        # NOTE: if this is giving a bunch of trouble switch to the "slower" code
        latent_s = torch.reshape(latent_s_batch, (-1, self.embed_dim, 21, 21))
        dec_latent_s = self.decode(latent_s)
        latent_ns = torch.reshape(latent_ns_batch, (-1, self.embed_dim, 21, 21))
        dec_latent_ns = self.decode(latent_ns)

        # TODO: no hard coding? and not sure if all this reshaping is giving me the right result, something to confirm
        state_batch = torch.reshape(dec_latent_s, (-1, self.history, 84, 84))
        next_state_batch = torch.reshape(dec_latent_ns, (-1, self.history, 84, 84))

        """Probably slower code done via chunking
        # Chunk the output into individual (latent) observations
        latent_s_list = torch.chunk(latent_s_batch, self.history, dim=1)
        latent_ns_list = torch.chunk(latent_ns_batch, self.history, dim=1)
        # Decode each latent state
        state_batch_list = []
        for latent_s in latent_s_list:
            state_batch_list.append(self.decode(latent_s))
        next_state_batch_list = []
        for latent_ns in latent_ns_list:
            next_state_batch_list.append(self.decode(latent_ns))

        print(latent_s_batch.size())  # TODO DELETE
        print(len(state_batch_list))  # TODO DELETE

        # Join the (decoded) observations into states and return
        state_batch = torch.cat(state_batch_list, dim=1)
        next_state_batch = torch.cat(next_state_batch_list, dim=1)
        """
        return state_batch, act_batch, rew_batch, next_state_batch

    def __len__(self) -> int:
        return self.buffer.size


def main():
    """For testing / debug purposes only """

    VqVaeModel(
        in_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=8,
        n_embed=512,
        device="cpu"
    )

    buf = CircularReplayBuffer(buffer_cap=15,
                               obs_shape=(2, 2),
                               history=3
                               )

    for _ob in range(20):
        cur_obs = torch.eye(2, dtype=torch.float32) * (_ob + 1)
        is_done = False

        if (_ob + 1) % 8 == 0:
            is_done = True

        buf.push(cur_obs, 1, 2, is_done)

        print(len(buf), buf.capacity)

    print(buf._done_buffer)
    # print(buf.get_valid_seq(9))

    # print(buf.encode_states([1, 8]))

    s, a, r, sp = buf.sample(3)
    print(s)
    print('===')
    print(sp)
    print(a)
    print(r)


if __name__ == "__main__":
    main()
