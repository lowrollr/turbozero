

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import torch
import math
from core.algorithms.evaluator import Evaluator, EvaluatorConfig

from core.env import Env
from core.utils.utils import rand_argmax_2d


@dataclass
class MCTSConfig(EvaluatorConfig):
    num_iters: int # number of MCTS iterations to run
    max_nodes: Optional[int] # maximum number of expanded nodes
    puct_coeff: float # C-value in PUCT formula
    dirichlet_alpha: float # magnitude of dirichlet noise
    dirichlet_epsilon: float # proportion of policy composed of dirichlet noise


# For now, this implementation of MCTS assumes that a model is used to evaluate leaf nodes rather than support rollouts
# TODO: implement a more general version that supports rollouts + other strategies
class MCTS(Evaluator):
    def __init__(self, env: Env, config: MCTSConfig, *args, **kwargs) -> None:
        super().__init__(env, config, *args, **kwargs)
        # search parameters
        # each evaluation traverses an edge of the search tree
        self.iters = config.num_iters
        # maximum nodes in the search tree, defaults to max_evals (essentially unbounded)
        self.max_nodes = config.max_nodes if config.max_nodes is not None else self.iters
        # C in the PUCT formula, controls exploration vs exploitation
        self.puct_coeff = config.puct_coeff
        # a positive value outside of the reward domain (e.g. 1e8) used to initialize the W-values of newly expanded nodes while preserving exploration order according to P-values
        self.very_positive_value = env.very_positive_value

        self.dirichlet_a = config.dirichlet_alpha
        self.dirichlet_e = config.dirichlet_epsilon

        self.parallel_envs = self.env.parallel_envs
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
            (self.parallel_envs, 2 + self.max_nodes, (4*self.policy_size)),
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
            (self.parallel_envs, self.max_nodes), dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the next empty node index for each env, used for expansion
        self.next_empty = torch.full_like(
            self.env_indices, 2, dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the current search depth (root = 1)
        self.depths = torch.ones(
            (self.parallel_envs,), dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the index (in self.nodes) of the current node for each env
        self.cur_nodes = torch.ones(
            (self.parallel_envs,), dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the indices of each node visited since leaving the root node
        self.visits = torch.zeros((self.parallel_envs, self.max_nodes),
                                  dtype=torch.int64, device=self.device, requires_grad=False)
        self.visits[:, 0] = 1

        self.reward_indices = self.build_reward_indices(env.num_players)
        self.dirilecht = torch.distributions.dirichlet.Dirichlet(torch.full((self.policy_size,), self.dirichlet_a, device=self.device, dtype=torch.float32, requires_grad=False))
        
        self.max_depths = torch.ones(
            (self.parallel_envs,), dtype=torch.int64, device=self.device, requires_grad=False
        )
        
        # stores the subtree 
        self.subtrees = torch.zeros((self.parallel_envs, self.max_nodes), dtype=torch.long, device=self.device, requires_grad=False) # index by master id
        self.parents = torch.zeros((self.parallel_envs, self.max_nodes), dtype=torch.long, device=self.device, requires_grad=False)
        self.depths_aranged = torch.arange(self.max_nodes, dtype=torch.int64, device=self.device, requires_grad=False)

    def build_reward_indices(self, num_players: int) -> torch.Tensor:
        num_repeats = math.ceil(self.max_nodes / num_players)
        return torch.tensor([1] + [0] * (num_players - 1), dtype=torch.bool, device=self.device).repeat(num_repeats).view(1, -1)
    
    def step_evaluator(self, actions, terminated):
        self.load_subtree(actions)
        self.reset_terminated_envs(terminated)

    def reset_terminated_envs(self, terminated: torch.Tensor) -> None:
        self.nodes[terminated].zero_()
        self.next_empty[terminated] = 2
        self.max_depths[terminated] = 1
        self.parents[terminated].zero_()

    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)
        self.nodes.zero_()
        self.reset_search()

    def reset_search(self) -> None:
        self.depths.fill_(1)
        self.cur_nodes.fill_(1)
        self.visits.zero_()
        self.visits[:, 0] = 1
        self.actions.zero_()

    def evaluate(self, evaluation_fn: Callable) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.reset_search()

        # save the root node so that we can reset the environment to this state when we reach a leaf node
        # get root node policy
        with torch.no_grad():
            policy_logits, initial_values = evaluation_fn(self.env)

        # set root node policy, apply dirilecht noise
        self.p_vals = (torch.softmax(policy_logits, dim=1) * (1 - self.dirichlet_e)) \
                + self.dirilecht.rsample(torch.Size((self.parallel_envs,))) * self.dirichlet_e # type: ignore
        

        saved = self.env.save_node()
        cur_players = self.env.cur_players.clone()

        for _ in range(self.iters):
            # choose next action with PUCT scores
            actions = self.choose_action()
        
            # make a step in the environment with the chosen actions
            self.env.step(actions)

            # check if the env is terminal
            terminated = self.env.is_terminal()

            # look up master index for each child node
            master_action_indices = self.next_indices[self.env_indices, actions]

            # if the node doesn't have an index yet (0 is null), a new node will be created
            unvisited = (master_action_indices == 0).long()

            # check if the next node will go out of bounds
            in_bounds = ~(((self.next_empty >= self.max_nodes) * unvisited).bool())

            # assign the leaf nodes the next empty indices (if there is room to expand the tree)
            master_action_indices += self.next_empty * in_bounds * unvisited

            master_action_indices_long = master_action_indices.long()

            # increment self.next_empty to reflect the new next empty index (if there is room to expand the tree)
            self.next_empty += in_bounds * unvisited

            # update the null values in the indices to reflect any new assigned indices
            self.nodes[self.env_indices, self.cur_nodes, self.i_start + actions] = master_action_indices

            # update the visits tensor to include the new node added to the path from the root
            self.visits[self.env_indices, self.depths] = master_action_indices_long
            self.actions[self.env_indices, self.depths - 1] = actions

            self.parents[self.env_indices, master_action_indices_long] = self.cur_nodes
            # cur nodes should now reflect the taken actions
            self.cur_nodes = master_action_indices_long

            # get policy and values for the new states from the model
            with torch.no_grad():
                policy_logits, values = evaluation_fn(self.env)

            # store the policy
            self.p_vals = torch.softmax(policy_logits, dim=1)

            is_leaf = (unvisited | (~in_bounds) | terminated ).bool()

            rewards = (self.env.get_rewards() * terminated) + (values.view(-1) * ~terminated)
            rewards = (rewards * (cur_players == self.env.cur_players)) + ((1-rewards) * (cur_players != self.env.cur_players))
            rewards.unsqueeze_(1)
            
            # propagate values and visit counts to nodes on the visited path (if the current node is a leaf)
            # (we only want to increment actual node visits, filter on visits > 0)
            valid = torch.roll(self.visits, -1, 1) > 0
            leaf_inc = valid * is_leaf.long().view(-1, 1)
            self.nodes[self.env_indices_expnd, self.visits, self.actions + self.n_start] += leaf_inc
            self.nodes[self.env_indices_expnd, self.visits, self.actions + self.w_start] += \
                (rewards * leaf_inc * self.reward_indices) + ((1-rewards) * leaf_inc * ~self.reward_indices)

            # update the depths tensor to reflect the current search depth for each environment   
            self.depths *= ~is_leaf
            self.depths += 1
            self.max_depths = torch.max(self.max_depths, self.depths)
            # zero out visits and actions if we've reached a leaf node
            self.visits[:, 1:] *= ~is_leaf.view(-1, 1)
            self.actions *= ~is_leaf.view(-1, 1)

            # reset to root node if we've reached a leaf node
            self.env.load_node(is_leaf, saved)
            self.cur_nodes *= ~is_leaf
            self.cur_nodes += is_leaf
            
            
            
        # return to the root node
        self.cur_nodes.fill_(1)
        self.env.load_node(self.cur_nodes.bool(), saved)
        # return visited counts at the root node
        max_inds = self.n_vals.max(dim=1).indices
        return self.n_vals, self.w_vals[self.env_indices, max_inds] / self.n_vals[self.env_indices, max_inds]

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

        return rand_argmax_2d(legal_puct_scores).flatten()
    
    def relax_subtrees(self):
        self.subtrees.zero_()
        self.subtrees += self.depths_aranged
        for _ in range(self.max_depths.max() + 1):
            parent_subtrees = self.subtrees[self.env_indices_expnd, self.parents]
            self.subtrees = (parent_subtrees * (parent_subtrees > 1)) + (self.subtrees * (parent_subtrees <= 1))
            
    def load_subtree(self, actions: torch.Tensor):
        self.relax_subtrees()
        # convert actions to master indices (N,)
        subtree_master_indices = self.nodes[self.env_indices, 1, self.i_start + actions]
        is_real = (self.nodes[self.env_indices, 1, self.i_start + actions] > 1)
        new_nodes = (self.subtrees == subtree_master_indices.view(-1, 1))
        # get master indices that belong to subtreee (others are zeroed out) (N, D)
        translation = new_nodes * is_real.view(-1, 1) * new_nodes.long().cumsum(dim=1) 
        nodes_copy = self.nodes.clone()
        self.nodes.zero_()
        # replace node indices with new indices (N, D)
        self.nodes[self.env_indices_expnd, translation] = nodes_copy[self.env_indices_expnd, self.depths_aranged * new_nodes]
        self.nodes[self.env_indices, :, self.i_start:self.i_end] = \
            translation[self.env_indices.view(-1, 1, 1), self.nodes[self.env_indices, :, self.i_start:self.i_end].long()].float()
        self.nodes[self.env_indices, :, self.n_start:self.n_end] *= self.nodes[self.env_indices, :, self.i_start:self.i_end] > 0
        translated_parents = translation[self.env_indices_expnd, self.parents].clone()
        self.parents.zero_()
        self.parents[self.env_indices_expnd, translation] = translated_parents
        self.next_empty = torch.amax(translation, dim=1) + 1
        self.next_empty.clamp_(min=2)
        self.max_depths -= 1
        self.max_depths.clamp_(min=1)
        
    