from typing import Optional, Tuple
import torch

class VectEnv:
    def __init__(self, 
            num_parallel_envs: int, 
            state_shape: torch.Size, 
            policy_shape: torch.Size, 
            value_shape: torch.Size, 
            device: torch.device, 
            is_stochastic: bool, 
            progression_batch_size: Optional[int] = None
    ):
        
        self.state_shape = state_shape
        self.policy_shape = policy_shape
        self.value_shape = value_shape

        self.states = torch.zeros((num_parallel_envs, *state_shape), dtype=torch.float32, device=device, requires_grad=False)
        self.invalid_mask = torch.zeros(num_parallel_envs, dtype=torch.bool, device=device, requires_grad=False)
        
        self.device = device
        self.is_stochastic = is_stochastic
        self.num_parallel_envs = num_parallel_envs
        
        self.progression_batch_size = progression_batch_size if progression_batch_size else num_parallel_envs


        # Tensors we re-use for indexing and sampling
        self.fws_cont = torch.ones(num_parallel_envs, dtype=torch.bool, device=device, requires_grad=False)
        self.fws_sums = torch.zeros(num_parallel_envs, dtype=torch.float32, device=device, requires_grad=False)
        self.fws_res = torch.zeros(num_parallel_envs, dtype=torch.int64, device=device, requires_grad=False)
        self.lnzero = torch.zeros(num_parallel_envs, dtype=torch.long, device=device, requires_grad=False)
        self.randn = torch.zeros(num_parallel_envs, dtype=torch.float32, device=device, requires_grad=False)
        self.fws_cont_batch = torch.ones(self.progression_batch_size , dtype=torch.bool, device=device, requires_grad=False)
        self.fws_sums_batch = torch.zeros(self.progression_batch_size , dtype=torch.float32, device=device, requires_grad=False)
        self.fws_res_batch = torch.zeros(self.progression_batch_size , dtype=torch.int64, device=device, requires_grad=False)
        self.lnzero_batch = torch.zeros(self.progression_batch_size , dtype=torch.long, device=device, requires_grad=False)
        self.randn_batch = torch.zeros(self.progression_batch_size , dtype=torch.float32, device=device, requires_grad=False)
        self.env_indices = torch.arange(num_parallel_envs, device=device, requires_grad=False)

    
    def reset(self, seed=None):
        raise NotImplementedError()
    
    def step(self, actions):
        self.push_actions(actions)
        if self.is_stochastic:
            # make step on legal states
            self.stochastic_step(torch.logical_not(self.invalid_mask))
        self.update_invalid_mask()
        return self.invalid_mask
    
    def update_invalid_mask(self):
        self.invalid_mask = self.is_terminal()

    def is_terminal(self):
        raise NotImplementedError()
    
    def push_actions(self, actions):
        raise NotImplementedError()
    
    def get_legal_actions(self):
        return torch.ones(self.num_parallel_envs, *self.policy_shape, dtype=torch.bool, device=self.device, requires_grad=False)

    def stochastic_step(self, mask=None) -> None:
        start_index = 0
        while start_index < self.num_parallel_envs:
            end_index = min(start_index + self.progression_batch_size, self.num_parallel_envs)
            states_batch = self.states[start_index:end_index]
            progs, probs = self.get_stochastic_progressions(states_batch)
            indices = self.fast_weighted_sample(probs, norm=True)
                
            if mask is not None:
                self.states[start_index:end_index] = torch.where(mask[start_index:end_index].view(self.num_parallel_envs, 1, 1, 1), progs[(self.env_indices[:end_index-start_index], indices)].unsqueeze(1), states_batch)
            else:
                self.states[start_index:end_index] = progs[(self.env_indices[:end_index-start_index], indices)].unsqueeze(1)

            start_index = end_index

    def get_stochastic_progressions(self, states_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
    
    def reset_invalid_states(self):
        raise NotImplementedError()

    def fast_weighted_sample(self, weights, norm=True, generator=None): # yields > 50% speedup over torch.multinomial for our use-cases!
        if norm:
            nweights = weights.div(weights.sum(dim=1, keepdim=True))
        else:
            nweights = weights

        num_samples = nweights.shape[0]
        num_categories = nweights.shape[1]

        # check if we are computing on a progression batch or on the full set of states
        if num_samples == self.progression_batch_size:
            self.fws_cont_batch.fill_(1)
            self.fws_sums_batch.zero_()
            self.fws_res_batch.zero_()
            self.randn_batch.uniform_(0, 1, generator = generator)
            self.lnzero_batch.zero_()
            conts, sums, res, rand_vals, lnzero = self.fws_cont_batch, self.fws_sums_batch, self.fws_res_batch, self.randn_batch, self.lnzero_batch
        else:
            self.fws_cont.fill_(1)
            self.fws_sums.zero_()
            self.fws_res.zero_()
            self.randn.uniform_(0, 1, generator = generator)
            self.lnzero.zero_()
            conts, sums, res, rand_vals, lnzero = self.fws_cont, self.fws_sums, self.fws_res, self.randn, self.lnzero

        for i in range(num_categories - 1):
            w_slice = nweights[:, i]
            sums.add_(w_slice)
            cont = rand_vals.ge(sums)
            res.add_(torch.logical_not(cont) * i * conts)
            conts.mul_(cont)
            lnzero = torch.max(lnzero, (w_slice != 0) * i)

        lnzero = torch.max(lnzero, (nweights[:, -1] != 0) * (num_categories - 1))

        res.add_(conts * lnzero)
        
        return res