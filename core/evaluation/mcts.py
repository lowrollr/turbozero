

from typing import Optional
import torch
from core.evaluation.evaluator import Evaluator

from core.vectenv import VectEnv
from .mcts_hypers import MCTSHypers

class VectorizedMCTS(Evaluator):
    def __init__(self, env: VectEnv, hypers: MCTSHypers) -> None:
        super().__init__(env, env.device, hypers)
        # search parameters
        # each evaluation traverses an edge of the search tree
        self.iters = hypers.num_iters
        # maximum depth of the search tree, defaults to max_evals (essentially unbounded)
        self.max_depth = hypers.max_depth if hypers.max_depth is not None else self.iters
        # C in the PUCT formula, controls exploration vs exploitation
        self.puct_coeff = hypers.puct_coeff
        # a positive value outside of the reward domain (e.g. 1e8) used to initialize the W-values of newly expanded nodes while preserving exploration order according to P-values
        self.very_positive_value = env.very_positive_value

        self.dirichlet_a = hypers.dirichlet_alpha
        self.dirichlet_e = hypers.dirichlet_epsilon

        self.parallel_envs = self.env.num_parallel_envs
        self.policy_size = self.env.policy_shape[-1]

        '''
        self.nodes holds the state of the search tree 
        = (NUM_ENVS * MAX_EVALS * (4 *self.policy_size))
        -> dim0: env index
        -> dim1: node index 
            -> 0: garbage/null/empty
            -> 1: search root
            -> 2->: children
        -> dim2: node info (m =self.policy_size)
            -> 0->m: action index -> node index mapping
            -> m->2m: action P-values
            -> 2m->3m: action N-values
            -> 3m->4m: action W-values

        (to make operations cleaner we use node index 0 as a 'garbage' index, and node index 1 as the root node)
        '''
        self.nodes = torch.zeros(
            (self.parallel_envs, 2 + self.iters, (4*self.policy_size)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        # helper tensors for indexing by env
        self.env_indices = self.env.env_indices
        self.env_indices_expnd = self.env_indices.view(-1, 1)

        # start/end indices for node info components, makes code cleaner
        self.i_start, self.i_end = 0, self.policy_size
        self.p_start, self.p_end = self.policy_size, 2*self.policy_size
        self.n_start, self.n_end = 2*self.policy_size, 3*self.policy_size
        self.w_start, self.w_end = 3*self.policy_size, 4*self.policy_size

        # MCTS
        # stores actions taken since leaving the root node, used for backpropagation
        self.actions = torch.zeros(
            (self.parallel_envs, self.max_depth), dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the next empty node index for each env, used for expansion
        self.next_empty = torch.full_like(
            self.env_indices, 2, dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the current search depth (root = 1)
        self.depths = torch.ones(
            (self.parallel_envs,), dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the index (in self.nodes) of the current node for each env
        self.cur_nodes = torch.ones(
            (self.parallel_envs,), dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the indices of each node's children
        self.visits = torch.zeros((self.parallel_envs, self.max_depth),
                                  dtype=torch.int64, device=self.device, requires_grad=False)
        self.visits[:, 0] = 1
        # tracks whether each env is currently at a leaf node
        self.is_leaf = torch.ones(
            (self.parallel_envs,), dtype=torch.bool, device=self.device, requires_grad=False)
        self.reward_indices = self.build_reward_indices(env.num_players)
        self.dirilecht = torch.distributions.dirichlet.Dirichlet(torch.full((self.policy_size,), self.dirichlet_a, device=self.device, dtype=torch.float32, requires_grad=False))

    def build_reward_indices(self, num_players: int) -> torch.Tensor:
        base_block = torch.tensor([1] + [0] * (num_players - 1), dtype=torch.bool, device=self.device)
        reward_indices = torch.zeros((num_players, self.max_depth), dtype=torch.bool, device=self.device)
        for i in range(num_players):
            rolled = torch.roll(base_block, i)
            repeated = rolled.repeat((self.max_depth // num_players) + 1)
            reward_indices[i] = repeated[:self.max_depth]
        return reward_indices

    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)
        self.nodes.zero_()
        self.reset_search()

    def evaluate(self, model: torch.nn.Module) -> torch.Tensor:
        self.reset_search()

        # save the root node so that we can reset the environment to this state when we reach a leaf node
        # get root node policy
        with torch.no_grad():
            policy_logits, _ = model(self.env.states)

        # set root node policy, apply dirilecht noise
        self.p_vals = (torch.softmax(policy_logits, dim=1) * (1 - self.dirichlet_e)) \
                + self.dirilecht.rsample(torch.Size((self.parallel_envs,))) * self.dirichlet_e # type: ignore
        

        self.env.save_node()

        for _ in range(self.iters):
            # choose next action with PUCT scores
            actions = self.choose_action()
            # look up master index for each child node
            master_action_indices = self.next_indices[self.env_indices, actions]
            # if the node doesn't have an index yet (0 is null), its a leaf node
            self.is_leaf = master_action_indices == 0 

            is_leaf_long = self.is_leaf.long()
            not_is_leaf_long = (~self.is_leaf).long()
            is_leaf_long_expnd = is_leaf_long.view(-1, 1)
            not_is_leaf_long_expnd = not_is_leaf_long.view(-1, 1)

            # assign the leaf nodes the next empty index
            master_action_indices += self.next_empty * is_leaf_long
            master_action_indices_long = master_action_indices.long()
            # increment self.next_empty to reflect the new next empty index
            self.next_empty += is_leaf_long
            # update the null values in the indices to reflect any new assigned indices
            next_indices = self.next_indices
            next_indices[self.env_indices, actions] = master_action_indices
            self.next_indices = next_indices

            # update the visits tensor to include the new node added to the path from the root
            self.visits[self.env_indices, self.depths] = master_action_indices_long
            self.actions[self.env_indices, self.depths - 1] = actions

            reward_indices = self.reward_indices[self.env.cur_players]

            # make a step in the environment with the chosen actions
            self.env.step(actions)

            # cur nodes should now reflect the taken actions
            self.cur_nodes = master_action_indices_long

            # get policy and values for the new states from the model
            with torch.no_grad():
                policy_logits, values = model(self.env.states)

            # store the policy
            self.p_vals = torch.softmax(policy_logits, dim=1)

            terminated = self.env.is_terminal()
            values = (self.env.get_rewards() * terminated) + (values.view(-1) * ~terminated)
            values.unsqueeze_(1)
            # propagate values and visit counts to nodes on the visited path (if the current node is a leaf)
            # (we only want to increment actual node visits, filter on visits > 0)
            valid = torch.roll(self.visits, -1, 1) > 0
            leaf_inc = valid * is_leaf_long_expnd
            self.nodes[self.env_indices_expnd, self.visits, self.actions + self.n_start] += leaf_inc
            self.nodes[self.env_indices_expnd, self.visits, self.actions + self.w_start] += (values * leaf_inc * reward_indices) + ((1-values) * leaf_inc * ~reward_indices)
            
            reset_to_root = terminated | self.is_leaf
            # update the depths tensor to reflect the current search depth for each environment
            self.depths *= ~reset_to_root
            self.depths += 1
            # zero out visits and actions if we've reached a leaf node
            self.visits[:, 1:] *= ~reset_to_root.view(-1, 1)
            self.actions *= ~reset_to_root.view(-1, 1)
            # reset to root node if we've reached a leaf node
            self.env.load_node(reset_to_root)
            self.cur_nodes *= ~reset_to_root
            self.cur_nodes += reset_to_root
        # return to the root node
        self.cur_nodes.fill_(1)
        self.env.load_node(self.cur_nodes.bool())
        # return visited counts at the root node
        return self.n_vals

    def reset_search(self) -> None:
        self.depths.fill_(1)
        self.cur_nodes.fill_(1)
        self.visits.zero_()
        self.visits[:, 0] = 1
        self.is_leaf.fill_(True)
        self.next_empty.fill_(2)
        self.actions.zero_()
        self.nodes.zero_()
    
    @property
    def next_indices(self) -> torch.Tensor:
        return self.nodes[self.env_indices, self.cur_nodes, self.i_start:self.i_end]

    @next_indices.setter
    def next_indices(self, values: torch.Tensor) -> None:
        self.nodes[self.env_indices, self.cur_nodes,
                   self.i_start:self.i_end] = values

    @property
    def p_vals(self) -> torch.Tensor:
        return self.nodes[self.env_indices, self.cur_nodes, self.p_start:self.p_end]

    @p_vals.setter
    def p_vals(self, values: torch.Tensor) -> None:
        self.nodes[self.env_indices, self.cur_nodes,
                   self.p_start:self.p_end] = values

    @property
    def n_vals(self) -> torch.Tensor:
        return self.nodes[self.env_indices, self.cur_nodes, self.n_start:self.n_end]

    @n_vals.setter
    def n_vals(self, values: torch.Tensor) -> None:
        self.nodes[self.env_indices, self.cur_nodes,
                   self.n_start:self.n_end] = values

    @property
    def w_vals(self) -> torch.Tensor:
        return self.nodes[self.env_indices, self.cur_nodes, self.w_start:self.w_end]

    @w_vals.setter
    def w_vals(self, values: torch.Tensor) -> None:
        self.nodes[self.env_indices, self.cur_nodes,
                   self.w_start:self.w_end] = values

    def choose_action(self) -> torch.Tensor:
        visits = self.n_vals
        zeros = (visits == 0)
        visits_augmented = visits + zeros

        q_values = self.w_vals / visits_augmented
        q_values += self.very_positive_value * zeros
        n_sum = visits.sum(dim=1, keepdim=True)
        puct_scores = q_values * \
            (self.puct_coeff * self.p_vals * torch.sqrt(n_sum + 1) / (1 + visits))

        legal_actions = self.env.get_legal_actions()

        legal_puct_scores = (puct_scores * legal_actions) - \
            (self.very_positive_value * (~legal_actions))

        return torch.argmax(legal_puct_scores, dim=1)
