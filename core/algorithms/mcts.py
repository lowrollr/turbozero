

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
        
        self.dirichlet_a = config.dirichlet_alpha
        self.dirichlet_e = config.dirichlet_epsilon

        self.parallel_envs = self.env.parallel_envs
        self.policy_size = self.env.policy_shape[-1]

        # garbage node + root node + other nodes
        self.total_slots = 2 + self.max_nodes 

        self.next_idx = torch.zeros(
            (self.parallel_envs, self.total_slots, self.policy_size),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )

        self.p_vals = torch.zeros(
            (self.parallel_envs, self.total_slots, self.policy_size),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.n_vals = torch.zeros(
            (self.parallel_envs, self.total_slots, self.policy_size),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )

        self.w_vals = torch.zeros(
            (self.parallel_envs, self.total_slots, self.policy_size),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        # helper tensors for indexing by env
        self.env_indices = self.env.env_indices
        self.env_indices_expnd = self.env_indices.view(-1, 1)

        # MCTS
        # stores actions taken since leaving the root node, used for backpropagation
        self.actions = torch.zeros(
            (self.parallel_envs, self.max_nodes + 1), dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the indices of each node visited since leaving the root node
        self.visits = torch.zeros((self.parallel_envs, 1 + self.max_nodes),
                                  dtype=torch.int64, device=self.device, requires_grad=False)
        self.visits[:, 0] = 1
        # stores the next empty node index for each env, used for expansion
        self.next_empty = torch.full_like(
            self.env_indices, 2, dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the current search depth (root = 1)
        self.depths = torch.ones(
            (self.parallel_envs,), dtype=torch.int64, device=self.device, requires_grad=False)
        # stores the index (in self.nodes) of the current node for each env
        self.cur_nodes = torch.ones(
            (self.parallel_envs,), dtype=torch.int64, device=self.device, requires_grad=False)
        
        
        self.reward_indices = self.build_reward_indices(env.num_players)
        assert self.dirichlet_a != 0
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.full((self.policy_size,), self.dirichlet_a, device=self.device, dtype=torch.float32, requires_grad=False))
        
        self.max_depths = torch.ones(
            (self.parallel_envs,), dtype=torch.int64, device=self.device, requires_grad=False
        )
        
        # stores the subtree 
        self.subtrees = torch.zeros((self.parallel_envs, self.total_slots), dtype=torch.long, device=self.device, requires_grad=False) # index by master id
        self.parents = torch.zeros((self.parallel_envs, self.total_slots), dtype=torch.long, device=self.device, requires_grad=False)
        self.slots_aranged = torch.arange(self.total_slots, dtype=torch.int64, device=self.device, requires_grad=False)

    def build_reward_indices(self, num_players: int) -> torch.Tensor:
        num_repeats = math.ceil((self.max_nodes + 1) / num_players)
        return torch.tensor([1] + [0] * (num_players - 1), dtype=torch.bool, device=self.device).repeat(num_repeats)[:self.max_nodes+1].view(1, -1)
    
    def step_evaluator(self, actions, terminated):
        self.load_subtree(actions)
        self.reset_evaluator_states(terminated)

    def reset_evaluator_states(self, evals_to_reset: torch.Tensor) -> None:
        self.next_idx[evals_to_reset] = 0
        self.n_vals[evals_to_reset] = 0
        self.w_vals[evals_to_reset] = 0
        self.p_vals[evals_to_reset] = 0
        self.next_empty[evals_to_reset] = 2
        self.max_depths[evals_to_reset] = 1
        self.parents[evals_to_reset] = 0

    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)
        self.next_idx.zero_()
        self.n_vals.zero_()
        self.w_vals.zero_()
        self.p_vals.zero_()
        self.next_empty.fill_(2)
        self.max_depths.fill_(1)
        self.parents.zero_()
        self.reset_search()

    def reset_search(self) -> None:
        self.depths.fill_(1)
        self.cur_nodes.fill_(1)
        self.visits.zero_()
        self.visits[:, 0] = 1
        self.actions.zero_()

    def choose_action(self) -> torch.Tensor:
        visits = self.n_vals[self.env_indices, self.cur_nodes]
        zero_visits = (visits == 0)
        visits_augmented = visits + zero_visits
        q_values = self.w_vals[self.env_indices, self.cur_nodes] / visits_augmented

        n_sum = visits.sum(dim=1, keepdim=True)
        probs = self.p_vals[self.env_indices, self.cur_nodes]
        puct_scores = q_values + (self.puct_coeff * probs * torch.sqrt(1 + n_sum) / (1 + visits))
        legal_actions = self.env.get_legal_actions()
        puct_scores = (puct_scores * legal_actions) + (torch.finfo(torch.float32).min * (~legal_actions))
        return torch.argmax(puct_scores, dim=1)
    
    def traverse(self, actions: torch.Tensor) -> torch.Tensor:
         # make a step in the environment with the chosen actions
        self.env.step(actions)

        # look up master index for each child node
        master_action_indices = self.next_idx[self.env_indices, self.cur_nodes, actions]

        # if the node doesn't have an index yet (0 is null), the node is unvisited
        unvisited = master_action_indices == 0

        # check if creating a new node will go out of bounds
        in_bounds = ~((self.next_empty >= self.total_slots) & unvisited)

        # assign new nodes to the next empty indices (if there is space)
        master_action_indices += self.next_empty * in_bounds * unvisited

        # increment self.next_empty to reflect the new next empty index 
        self.next_empty += 1 * in_bounds * unvisited

        # map action to child idx in parent node
        self.next_idx[self.env_indices, self.cur_nodes, actions] = master_action_indices

        # update visits, actions to reflect the path taken from the root
        self.visits[self.env_indices, self.depths] = master_action_indices
        self.actions[self.env_indices, self.depths - 1] = actions

        # map master child idx to master parent idx
        self.parents[self.env_indices, master_action_indices] = self.cur_nodes

        # cur nodes should now reflect the taken actions
        self.cur_nodes = master_action_indices

        return unvisited
        

    def evaluate(self, evaluation_fn: Callable) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.reset_search()

        # get root node policy
        with torch.no_grad():
            policy_logits, _ = evaluation_fn(self.env)

        # set root node policy, apply dirichlet noise
        legal_actions = self.env.get_legal_actions()
        policy_logits = (policy_logits * legal_actions) + (torch.finfo(torch.float32).min * (~legal_actions))

        self.p_vals[self.env_indices, self.cur_nodes] = (torch.softmax(policy_logits, dim=1) * (1 - self.dirichlet_e)) \
                + (self.dirichlet.rsample(torch.Size((self.parallel_envs,))) * self.dirichlet_e)
        
        # save root node, so that we can load it again when a leaf node is reached
        saved = self.env.save_node()

        # store player id of current player at root node
        cur_players = self.env.cur_players.clone()

        for _ in range(self.iters):
            # choose next action with PUCT scores
            actions = self.choose_action()
        
            unvisited = self.traverse(actions)

            # get (policy distribution, evaluation) from evaluation function
            with torch.no_grad():
                policy_logits, values = evaluation_fn(self.env)

            legal_actions = self.env.get_legal_actions()

            policy_logits = (policy_logits * legal_actions) + (torch.finfo(torch.float32).min * (~legal_actions))
            # store the policy
            self.p_vals[self.env_indices, self.cur_nodes] = torch.softmax(policy_logits, dim=1)

            # check if the env has terminated
            terminated = self.env.is_terminal()

            # rewards are actual env rewards if env is terminated, else they are the value estimates
            rewards = (self.env.get_rewards() * terminated) + (values.view(-1) * ~terminated)
            rewards = (rewards * (cur_players == self.env.cur_players)) + ((1-rewards) * (cur_players != self.env.cur_players))
            rewards.unsqueeze_(1)

            # we have reached a leaf node if:
            #  -> the current node is unvisited
            #   OR
            #  -> the current node is terminal
            is_leaf = unvisited | terminated
            
            # propagate values and visit counts to nodes on the visited path (if the current node is a leaf)
            # (we only want to increment actual node visits, filter on visits > 0)
            valid = torch.roll(self.visits, -1, 1) > 0
            valid[:,-1] = 0
            leaf_inc = valid * is_leaf.long().view(-1, 1)
            self.n_vals[self.env_indices_expnd, self.visits, self.actions] += leaf_inc
            self.w_vals[self.env_indices_expnd, self.visits, self.actions] += (rewards * leaf_inc * self.reward_indices) + ((1-rewards) * leaf_inc * ~self.reward_indices)

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

        # reload all envs to the root node
        self.env.load_node(self.cur_nodes.bool(), saved)
        
        # return visited counts at the root node
        max_inds = self.n_vals[self.env_indices, self.cur_nodes].argmax(dim=1)
        max_q_vals = self.w_vals[self.env_indices, self.cur_nodes, max_inds] / self.n_vals[self.env_indices, self.cur_nodes, max_inds]
        n_vals_sum = self.n_vals[self.env_indices, self.cur_nodes].sum(dim=1, keepdim=True)
        return self.n_vals[self.env_indices, self.cur_nodes] / n_vals_sum, max_q_vals
    
    def propogate_root_subtrees(self):
        self.subtrees.zero_()
        self.subtrees += self.slots_aranged
        self.parents[:, 0] = 0
        for _ in range(self.max_depths.max() + 1):
            parent_subtrees = self.subtrees[self.env_indices_expnd, self.parents]
            self.subtrees = (parent_subtrees * (parent_subtrees > 1)) + (self.subtrees * (parent_subtrees <= 1))
            
    def load_subtree(self, actions: torch.Tensor):
        self.propogate_root_subtrees()
        # convert actions to master indices (N,)
        subtree_master_indices = self.next_idx[self.env_indices, 1, actions]
        is_real = subtree_master_indices > 1
        new_nodes = (self.subtrees == subtree_master_indices.view(-1, 1))
        # get master indices that belong to subtreee (others are zeroed out) (N, D)
        translation = new_nodes * is_real.view(-1, 1) * new_nodes.long().cumsum(dim=1)
        # replace node indices with new indices (N, D)

        old_subtree_idxs = self.slots_aranged * new_nodes

        self.next_empty = torch.amax(translation, dim=1) + 1
        erase = self.slots_aranged * (self.slots_aranged >= self.next_empty.view(-1, 1))
        self.next_empty.clamp_(min=2)
        
        self.w_vals[self.env_indices_expnd, translation] = self.w_vals[self.env_indices_expnd, old_subtree_idxs]
        self.w_vals[self.env_indices_expnd, erase] = 0
        
        self.n_vals[self.env_indices_expnd, translation] = self.n_vals[self.env_indices_expnd, old_subtree_idxs]
        self.n_vals[self.env_indices_expnd, erase] = 0
        
        self.p_vals[self.env_indices_expnd, translation] = self.p_vals[self.env_indices_expnd, old_subtree_idxs]
        self.p_vals[self.env_indices_expnd, erase] = 0
        
        self.next_idx[self.env_indices_expnd, translation] = translation[self.env_indices.view(-1, 1, 1), self.next_idx]
        self.next_idx[self.env_indices_expnd, erase] = 0

        self.parents[self.env_indices_expnd, translation] = translation[self.env_indices_expnd, self.parents]
        self.parents[self.env_indices_expnd, erase] = 0

        
        self.max_depths -= 1
        self.max_depths.clamp_(min=1)
        
    