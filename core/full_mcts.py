

from typing import Optional
import torch

from core.vectenv import VectEnv


class VectorizedMCTS:
    def __init__(self, env: VectEnv, max_evals: int, puct_coeff: float, very_positive_value: float = 1e8, max_depth: Optional[int] = None) -> None:
        self.env = env
        self.max_evals = max_evals
        self.puct_coeff = puct_coeff
        self.vpv = very_positive_value
        self.max_depth: int = max_depth if max_depth is not None else max_evals 

        device = self.env.device
        parallel_envs = self.env.num_parallel_envs
        policy_size = self.env.policy_shape[-1]
        self.env_indices = self.env.env_indices
        self.env_indices_expnd = self.env_indices.view(-1,1)
        self.env.reset()

        '''
        self.nodes:  
        -> dim0: env index
        -> dim1: node index 
            -> 0: garbage/null/empty
            -> 1: search root
            -> 2->: children
        -> dim2: node info
            -> 0-m: action index -> node index mapping
            -> m-2m: action P-values
            -> 2m-3m: action N-values
            -> 3m-4m: action W-values
        '''
        self.nodes = torch.zeros(
            (parallel_envs, 2 + max_evals, (4*policy_size)),
            dtype = torch.float32,
            device = device,
            requires_grad = False
        )
        self.i_start, self.i_end = 0, policy_size
        self.p_start, self.p_end = policy_size, 2*policy_size
        self.n_start, self.n_end = 2*policy_size, 3*policy_size
        self.w_start, self.w_end = 3*policy_size, 4*policy_size
        
        self.actions = torch.zeros((parallel_envs, max_evals), dtype=torch.int64, device=device, requires_grad=False)
        self.next_empty = torch.full_like(self.env_indices, 2, dtype=torch.int64, device=device, requires_grad=False)
        self.depths = torch.ones((parallel_envs,), dtype=torch.int64, device=device, requires_grad=False)
        self.cur_nodes = torch.ones((parallel_envs,), dtype=torch.int64, device=device, requires_grad=False)
        self.visits = torch.zeros((parallel_envs, self.max_depth), dtype=torch.int64, device=device, requires_grad=False)
        self.visits[:, 0] = 1
        self.is_leaf = torch.ones((parallel_envs,), dtype=torch.bool, device=device, requires_grad=False)


    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)
        self.reset_search()

    def reset_search(self) -> None:
        self.depths.fill_(1)
        self.cur_nodes.fill_(1)
        self.visits.zero_()
        self.visits[:, 0] = 1
        self.nodes.zero_()
        self.is_leaf.fill_(True)
        self.next_empty.fill_(2)
        self.actions.zero_()

    @property
    def next_indices(self) -> torch.Tensor:
        return self.nodes[self.env_indices, self.cur_nodes, self.i_start:self.i_end]
    
    @next_indices.setter
    def next_indices(self, values: torch.Tensor) -> None:
        self.nodes[self.env_indices, self.cur_nodes, self.i_start:self.i_end] = values

    @property
    def p_vals(self) -> torch.Tensor:
        return self.nodes[self.env_indices, self.cur_nodes, self.p_start:self.p_end]
    
    @p_vals.setter
    def p_vals(self, values: torch.Tensor) -> None:
        self.nodes[self.env_indices, self.cur_nodes, self.p_start:self.p_end] = values
    
    @property
    def n_vals(self) -> torch.Tensor:
        return self.nodes[self.env_indices, self.cur_nodes, self.n_start:self.n_end]
        
    @n_vals.setter
    def n_vals(self, values: torch.Tensor) -> None:
        self.nodes[self.env_indices, self.cur_nodes, self.n_start:self.n_end] = values

    @property
    def w_vals(self) -> torch.Tensor:
        return self.nodes[self.env_indices, self.cur_nodes, self.w_start:self.w_end]

    @w_vals.setter
    def w_vals(self, values: torch.Tensor) -> None:
        self.nodes[self.env_indices, self.cur_nodes, self.w_start:self.w_end] = values

    def legal_actions(self) -> torch.Tensor:
        return self.p_vals > 0
    
    def propagate_values(self, values: torch.Tensor) -> None:
        pass

    def choose_action(self) -> torch.Tensor:
        visits = self.n_vals
        zeros = (visits == 0)
        visits_augmented = visits + zeros
        
        q_values = self.w_vals / visits_augmented
        q_values += self.vpv * zeros
        n_sum = visits.sum(dim=1, keepdim=True)
        puct_scores = q_values * (self.puct_coeff * self.p_vals * torch.sqrt(n_sum + 1) / (1 + visits))

        legal_actions = self.legal_actions()

        legal_puct_scores = (puct_scores * legal_actions) - (self.vpv * (~legal_actions))

        return torch.argmax(legal_puct_scores, dim=1)

    def explore(self, model: torch.nn.Module) -> torch.Tensor:
        self.reset_search()
        # save the root node so that we can reset the environment to this state when we reach a leaf node
        self.env.save_node()

        # get root node policy
        with torch.no_grad():
            policy_logits, _ = model(self.env.states)

        # set root node policy 
        self.p_vals = (torch.softmax(policy_logits, dim=1) * self.env.get_legal_actions())

        for i in range(self.max_evals):
            # choose next action with PUCT scores
            actions = self.choose_action()
            # look up master index for each child node
            master_action_indices = self.next_indices[self.env_indices, actions]
            # if the node doesn't have an index yet (0 is null), its a leaf node
            self.is_leaf = master_action_indices == 0
            # assign the leaf nodes the next empty index
            master_action_indices += self.next_empty * self.is_leaf
            # increment self.next_empty to reflect the new next empty index
            self.next_empty += 1 * self.is_leaf
            # update the null values in the indices to reflect any new assigned indices
            next_indices = self.next_indices
            next_indices[self.env_indices, actions] = master_action_indices
            self.next_indices = next_indices
            # update the visits tensor to include the new node added to the path from the root
            self.visits[self.env_indices, self.depths] = master_action_indices.long()
            self.actions[self.env_indices, self.depths - 1] = actions
            # make a step in the environment with the chosen actions
            self.env.step(actions)
            # cur nodes should now reflect the taken actions
            self.cur_nodes = master_action_indices.long()
            # get policy and values for the new states from the model
            with torch.no_grad():
                policy_logits, values = model(self.env.states)
            # store the policy, zeroing out illegal moves
            self.p_vals = torch.softmax(policy_logits, dim=1) * self.env.get_legal_actions()
            # propagate values and visit counts to nodes on the visited path (if the current node is a leaf)
            # (we only want to increment actual node visits, filter on visits > 0)
            valid = torch.roll(self.visits, -1, 1) > 0
            self.nodes[self.env_indices_expnd, self.visits, self.actions + self.n_start] += (1 * valid * self.is_leaf.view(-1, 1))
            self.nodes[self.env_indices_expnd, self.visits, self.actions + self.w_start ] += (values * valid * self.is_leaf.view(-1, 1))
            # update the depths tensor to reflect the current search depth for each environment
            self.depths *= (~self.is_leaf).long()
            self.depths += 1 
            # zero out visits if we've reached a leaf node
            self.visits[:, 1:] *= 1 * (~self.is_leaf).view(-1, 1)
            self.actions *= 1 * (~self.is_leaf).view(-1, 1)
            # reset to root node if we've reached a leaf node
            self.env.load_node(self.env_indices * self.is_leaf)
            self.cur_nodes *= (~self.is_leaf).long()
            self.cur_nodes += self.is_leaf.long()
        # return to the root node
        self.cur_nodes.fill_(1)
        self.env.load_node(self.cur_nodes)
        # return visited counts at the root node
        return self.n_vals
            



        