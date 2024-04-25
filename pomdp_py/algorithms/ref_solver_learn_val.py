"""
Add updated comment here.
"""

from pomdp_py.framework.basics import Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel,\
    sample_generative_model
from pomdp_py.framework.planner import Planner
from pomdp_py.representations.distribution.particles import Particles
from pomdp_py.representations.belief.particles import particle_reinvigoration
from pomdp_py.utils import typ
import copy
import time
import random
import math
import numpy as np
from tqdm import tqdm
import json
import os
import sys

import torch as T


# TODO: Tidy up argument list for RefSolver class.
# TODO: Tidy up node parameters.
# TODO: Check consistent discount_factor is being used throughout the solver.

# TODO: Move to some kind of template file.
EPSILON = 1e-20

n = 20

# Function to perform one-hot encoding for a given state
def one_hot_encode_state(state):
    # Map the 2D state to a unique index

    index = (state[0] - 1) * n + (state[1] - 1)
    
    # Initialize the one-hot encoded vector
    encoded_vector = np.zeros(n * n)
    
    # Set the element at the index to 1
    encoded_vector[index] = 1
    
    return encoded_vector

class TreeNodeRef:
    def __init__(self):
        self.children = {}
    def __getitem__(self, key):
        return self.children.get(key, None)
    def __setitem__(self, key, value):
        self.children[key] = value
    def __contains__(self, key):
        return key in self.children

class QNodeRef(TreeNodeRef):
    def __init__(self, num_visits, r_est):
        """
        `history_action`: a tuple ((a,o),(a,o),...(a,)). This is only
            used for computing hashes
        """
        self.num_visits = num_visits
        self.r_est = r_est
        self.children = {} # o _. VNode
    def __str__(self):
        return typ.red("QNode") + "(N: %d, R est: %d | %s)" % (self.num_visits, self.r_est,
                                                    str(self.children.keys()))
    
    def __repr__(self):
        return self.__str__()

    def print_children_z_values(self):
        for observation in self.children:
            print("   observation %s: %.3f" % (str(observation), self[observation].z_value))

class VNodeRef(TreeNodeRef):
    def __init__(self, num_visits, z_value, r_est, G, **kwargs):
        self.num_visits = num_visits
        self.z_value = z_value
        self.r_est = r_est
        self.G = G
        self.children = {}  # a -> QNode
    def __str__(self):
        return typ.green("VNode") + "(N: %d, V: %.3f, R est: %.3f, G: %.3f | %s)"\
                                    % (self.num_visits,
                                        math.log(self.z_value),
                                        self.r_est,
                                        self.G,
                                        str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

class RootVNodeRef(VNodeRef):
    def __init__(self, num_visits, z_value, r_est, G, history):
        VNodeRef.__init__(self, num_visits, z_value, r_est, G)
        self.history = history
    @classmethod
    def from_vnode_ref(cls, vnode, history):
        rootnode = RootVNodeRef(vnode.num_visits, 
                                vnode.z_value, 
                                vnode.r_est,
                                vnode.G,
                                history)
        rootnode.children = vnode.children
        return rootnode

class VNodeRefParticles(VNodeRef):
    """Maintains particle belief"""
    def __init__(self, num_visits, z_value, r_est, G, belief=Particles([])):
        self.num_visits = num_visits
        self.z_value = z_value
        self.r_est = r_est
        self.G = G
        self.belief = belief
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(N: %d, V: %.3f, R est: %.3f, G: %.3f, no. particles: %d | %s)"\
            % (
                self.num_visits, 
                math.log(self.z_value),
                self.r_est,
                self.G,
                len(self.belief),
                str(self.children.keys())
              )
    def __repr__(self):
        return self.__str__()

class RootVNodeRefParticles(RootVNodeRef):
    def __init__(self, num_visits, z_value, r_est, G, history, belief=Particles([])):
        RootVNodeRef.__init__(self, num_visits, z_value, r_est, G, history)
        self.belief = belief
    @classmethod
    def from_vnode_ref(cls, vnode, history):
        rootnode = RootVNodeRefParticles(vnode.num_visits, vnode.z_value, vnode.r_est,\
                                      vnode.G, history, belief=vnode.belief)
        rootnode.children = vnode.children
        return rootnode

class RefSolverLearn(Planner):
    # TODO: Update the comment.
    """Implements the solver from Kim and Kurniawati.

    Args:
        max_depth (int): Depth of the search tree. Default: 5.
        max_rollout_depth (int): Depth of the rollout. Default: 20. 
        planning_time (float), amount of time given to each planning step (seconds). Default: -1.
            if negative, then planning terminates when number of simulations `num_sims` reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        num_sims (int): Number of simulations for each planning step. If negative,
            then will terminate when planning_time is reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        show_progress (bool): True if print a progress bar for simulations.
        pbar_update_interval (int): The number of simulations to run after each update of the progress bar,
            Only useful if show_progress is True; You can set this parameter even if your stopping criteria
            is time.
    """

    def __init__(self,
                 learning_agent,
                 max_depth=5, max_rollout_depth=20, planning_time=-1., num_sims=-1,
                 discount_factor=0.95,
                 exploration_const=0.2, # currently used for naive sampler.
                 rew_shift=-100, # shift size for reward
                 rew_scale=1, # scaling factor for reward
                 fully_obs_generator=None, # the algorithm used to generate a fully observed policy.
                 fully_obs_policy=dict(), # include if a fully observed policy has already been generated offline.
                 num_visits_init=0, z_val_init=1.0, r_est_init=0.0, G_init=0.0,
                 show_progress=False, pbar_update_interval=5):
        
        if exploration_const < 0 and exploration_const >= 1:
            raise ValueError("RefSolver expects an exploration constant that is a probability in [0, 1).")

        # bespoke params
        ###########################################
        self._z_val_init = z_val_init
        self._r_est_init = r_est_init
        self._G_init = G_init
        self._u_opt = None # to store the estimated optimal stochastic policy after planning.
        self._rew_shift = rew_shift
        self._rew_scale = rew_scale
        self._fully_obs_generator = fully_obs_generator
        self._est_fully_obs_policy = None # to store the estimated fully observed policy at a given belief.
        self._fully_obs_policy = fully_obs_policy # can be used to store the fully observed policy for DP.
        ###########################################

        self.learning_agent = learning_agent
        self._max_depth = max_depth # i.e. max_depth for planning
        self._max_rollout_depth = max_rollout_depth # i.e. max depth for rollout
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._num_visits_init = num_visits_init
        self._discount_factor = discount_factor
        self._exploration_const = exploration_const
        
        self._show_progress = show_progress
        self._pbar_update_interval = pbar_update_interval
        
        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

        self.simulation_count = 0
        self.simulation_logs = {}

    @property
    def update_agent_belief(self):
        """True if planner's update function also updates the agent's belief."""
        return True
    
    @property
    def last_num_sims(self):
        """Returns the number of simulations run for the last `plan` call."""
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) taken for 
        the last `plan` call to run."""
        return self._last_planning_time

    def plan(self, agent):

        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented using particles.\n"\
                            "This is incompatible with the current solver.\n"\
                            "Please convert it to particles.")

        self._agent = agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, time_taken, sims_count, u_opt, est_fully_obs_policy = self._search()
        
        print("Root:", agent.tree)
        print("V", math.log(agent.tree.z_value))
        
        if any([np.isnan(p) for p in u_opt.values()]):
            action = self._fully_obs_policy_dp(self._agent.sample_belief())
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        self._u_opt = u_opt
        self._est_fully_obs_policy = est_fully_obs_policy
        return action
    
    def update(self, agent, real_action, real_observation,
            state_transform_func=None):
        """
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.
        `state_transform_func`: Used to add artificial transform to states during
            particle reinvigoration. Signature: s -> s_transformed
        """
        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented using particles.\n"\
                            "This is incompatible with the current solver.\n"\
                            "Please convert it to particles.")
        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have a tree. Have you planned yet?")
            return
        
        invigorated_belief = particle_reinvigoration(agent.belief,
                                                     len(agent.init_belief.particles))
        
        agent.set_belief(invigorated_belief)
        
        next_belief = Particles([])
        for state in agent.belief:
            next_state, observation, reward, nsteps = sample_generative_model(
                                                self._agent, state, real_action)
            if observation.__eq__(real_observation):
                next_belief.add(next_state)
        
        if agent.tree[real_action][real_observation] is None:
            # # Never anticipated the real_observation. No reinvigoration can happen.
            # raise ValueError("Particle deprivation.")
    
            agent.tree[real_action][real_observation] = RootVNodeRefParticles(
                                self._num_visits_init,
                                self._z_val_init,
                                self._r_est_init,
                                self._G_init,
                                self._agent.history + ((real_action, real_observation),),
                                belief=next_belief)
            
        else:
            agent.tree[real_action][real_observation].belief = next_belief
                
        ##################################################################################
        # Update the tree; Reinvigorate the tree's belief and use it
        # as the updated belief for the agent.
        agent.tree = RootVNodeRefParticles.from_vnode_ref(agent.tree[real_action][real_observation],
                                                    agent.history)
        # tree_belief = agent.tree.belief
        agent.set_belief(particle_reinvigoration(next_belief,
                                                    len(agent.init_belief.particles),
                                                    state_transform_func=state_transform_func))
            
        agent.tree.belief = copy.deepcopy(agent.belief)
        ##################################################################################
    
    # def update(self, agent, real_action, real_observation,
    #         obs_support=None, state_transform_func=None):
        
    #     """
    #     Assume that the agent's history has been updated after taking real_action
    #     and receiving real_observation.
    
    #     `state_transform_func`: Used to add artificial transform to states during
    #         particle reinvigoration. Signature: s -> s_transformed
            
    #     `obs_support`: A mapping that takes as input a real_observation and
    #         returns a list of (next) states that are consistent with the observation.
    #     """
    
    #     if not isinstance(agent.belief, Particles):
    #         raise TypeError("Agent's belief is not represented using particles.\n"\
    #                         "This is incompatible with the current solver.\n"\
    #                         "Please convert it to particles.")
    
    #     if not hasattr(agent, "tree") or agent.tree is None:
    #         print("Warning: agent does not have a tree. Have you planned yet?")
    #         return
    
    #     if agent.tree[real_action][real_observation] is None:
    #         # Never anticipated the real_observation. No reinvigoration can happen.
    #         if obs_support is None:
    #             raise ValueError("Particle deprivation.")
    #         else:
    #             possible_states = Particles(obs_support(real_observation))
            
    #         next_belief = Particles([])
    #         particles = 0
    #         timer = 0
    #         while particles < len(agent.init_belief.particles):
                
    #             if timer > 1e5:
    #                 raise ValueError("Particle deprivation.")
                    
    #             state = possible_states.random()
    #             next_state, observation, reward, nsteps = sample_generative_model(
    #                                                 self._agent, state, real_action)
    #             if observation.__eq__(real_observation):
    #                 next_belief.add(next_state)
    #                 particles += 1
    #             timer += 1
                
    #         agent.tree[real_action][real_observation] = RootVNodeRefParticles(
    #                             self._num_visits_init,
    #                             self._z_val_init,
    #                             self._r_est_init,
    #                             self._G_init,
    #                             self._agent.history + ((real_action, real_observation),),
    #                             belief=next_belief)
                    
    #     ##################################################################################
    #     # Update the tree; Reinvigorate the tree's belief and use it
    #     # as the updated belief for the agent.
        
    #     agent.tree = RootVNodeRefParticles.from_vnode_ref(agent.tree[real_action][real_observation],
    #                                                     agent.history)
        
    #     tree_belief = agent.tree.belief
    #     agent.set_belief(particle_reinvigoration(tree_belief,
    #                                                 len(agent.init_belief.particles),
    #                                                 state_transform_func=state_transform_func))
            
    #     agent.tree.belief = copy.deepcopy(agent.belief)
    #     ##################################################################################
    

    def clear_agent(self):
        self._agent = None
        self._last_num_sims = -1

    def _add_qnode(self, vnode, action):
        
        if action not in self._agent.all_actions:
            raise ValueError("Invalid action.")
            
        if vnode[action] is None:
            history_action_node = QNodeRef(self._num_visits_init,
                                        self._r_est_init)
            vnode[action] = history_action_node

    def _search(self):
        sims_count = 0
        time_taken = 0
        stop_by_sims = self._num_sims > 0

        if self._show_progress:
            if stop_by_sims:
                total = int(self._num_sims)
            else:
                total = self._planning_time
            pbar = tqdm(total=total)

        start_time = time.time()
        
        while True:
            ## Note: the tree node with () history will have
            ## the init belief given to the agent.
            state = self._agent.sample_belief()

            # # Get estimated future reward of starting state and write to file
            # state_tensor = T.tensor(state.position, dtype=T.float32).to(self.learning_agent.critic.device)
            # val = self.learning_agent.critic(state_tensor)
            # val = T.squeeze(val).item()
            # # Write estimated future expected reward to log file
            # log_file.write(f"Estimated future expected reward for sim {sims_count}: {val}\n")

            self._simulate(state, 
                        self._agent.history,
                        self._agent.tree,
                        0,
                        exploration_const=self._exploration_const)

            self.simulation_count += 1
            
            # Get true episode reward and write to same file 
            sims_count +=1
            time_taken = time.time() - start_time

            if self._show_progress and sims_count % self._pbar_update_interval == 0:
                if stop_by_sims:
                    pbar.n = sims_count
                else:
                    pbar.n = time_taken
                pbar.refresh()

            if stop_by_sims:
                if sims_count >= self._num_sims:
                    break
            else:
                if time_taken > self._planning_time: 
                    if self._show_progress:
                        pbar.n = self._planning_time
                        pbar.refresh()
                    break

        if self._show_progress:
            pbar.close()
            
        u_opt = self._get_u_opt(self._agent.tree)
        est_fully_obs_policy = Particles([self._fully_obs_policy_dp(s) for s in self._agent.tree.belief.particles])   
        print("Stochastic policy: ", u_opt)
        action = random.choices(list(u_opt.keys()), list(u_opt.values()))[0]
        return action, time_taken, sims_count, u_opt, est_fully_obs_policy
        
    def _rollout(self, state, history, depth, exploration_const):
        
        adj_reward = lambda state, action : (self._agent._reward_model.reward_func(state, action)\
                                             - self._rew_shift) * self._rew_scale
        
        action = self._fully_obs_policy_dp(state)
        #TODO: This part makes it specific to a problem with goal states.
        if state.terminal or depth > self._max_rollout_depth:
            # Choose an arbitrary action as goal state is absorbing.
            return adj_reward(state, action) 
        
        next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
        reward = (reward - self._rew_shift) * self._rew_scale
        
        return reward + self._discount_factor * \
            self._rollout(next_state,
                          history + ((action, observation),),
                          depth+1,
                          exploration_const)
    
    def _simulate(self, state, history, root, depth, exploration_const):
        """Performs forward search using the reference policy and backs up
        each path using the analytic Bellman equation."""

        if state.terminal:

            action = self._fully_obs_policy_dp(state)
        
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)

            one_hot_obs = one_hot_encode_state(next_state.position)

            observation_tensor = T.tensor(one_hot_obs, dtype=T.float32).to(self.learning_agent.critic.device)
        

            # Pass the tensor to the critic and handle possible exceptions
            try:
                val = self.learning_agent.critic(observation_tensor)
                val = T.squeeze(val).item()  # Convert to Python scalar and ensure it's a float
            except Exception as e:
                print("An error occurred during critic evaluation:", e)

            # Now call remember, ensuring all types match and are correctly handled
            try:
                self.learning_agent.remember(observation_tensor.cpu().numpy(), action, val, reward, next_state.terminal)
            except Exception as e:
                print("An error occurred while remembering the state:", e)

            self.learning_agent.learn()
            val = math.exp(self._rollout(state, history, depth, exploration_const))

            # Write final rollout value to the simulation logs
            simulation_data = {
                "step": "Terminal Value",
                "rollout_value": val
            }

            simulation_id = self.simulation_count

            # try:
            #     self.simulation_logs[simulation_id]["steps"].insert(0, simulation_data)
            #     self._write_simulation_logs_to_file()
            # except Exception as e:
            #     print("An error occurred while writing the final rollout value:", e)

            return val

        # Observation:  float32
        # Action:  int64
        # Val:  <class 'float'>
        # Reward:  <class 'float'>
        # Done:  <class 'bool'>
        
        # Check if the current depth exceeds maximum depth and handle learning
        if depth > self._max_depth:
            
            self.learning_agent.learn()

            one_hot_state = one_hot_encode_state(state.position)

            state_tensor = T.tensor(one_hot_state, dtype=T.float32).to(self.learning_agent.critic.device)

            val = self.learning_agent.critic(state_tensor)
            val = T.squeeze(val).item()  # Ensures it's a Python float

            # val = val * 5.0

            final_val = math.exp(val)

            # Write final rollout value to the simulation logs
            simulation_data = {
                "step": "Rollout Value",
                "rollout_value": final_val
            }

            simulation_id = self.simulation_count

            # for key in self.simulation_logs.keys():
            #     print(key)

            # try:
            #     self.simulation_logs[simulation_id]["steps"].insert(0, simulation_data)
            # except Exception as e:
            #     print("An error occurred while inserting the final rollout value:", e)
            #     print(self.simulation_logs)
            #     sys.exit()

            # try:
            #      self._write_simulation_logs_to_file()
            # except Exception as e:
            #     print("An error occurred while writing the final rollout value:", e)
            
            return final_val
            # return math.exp(self._rollout(state, history, depth, exploration_const))
        
        adj_reward = lambda state, action : (self._agent._reward_model.reward_func(state, action)\
                                             - self._rew_shift) * self._rew_scale

        if root is None:
            if self._agent.tree is None:
                root = self._VNodeRef(agent=self._agent, root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNodeRef()
        
        root.belief.add(state)
        root.num_visits += 1

        # TODO: Perhaps this can be improved.        
        # ********** Naive Action Sampler  **********
        # fo_action = self._fully_obs_policy_dp(state)
        # exp_const = 0 if self._num_sims < 0 else 1 - root[fo_action].num_visits / self._num_sims
        
        coin = random.choices([True, False],[1-exploration_const, exploration_const],k=1)[0]
        action = self._fully_obs_policy_dp(state) if coin else random.sample(list(self._agent.all_actions), 1)[0]
        # ********************************************
                
        next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)


        # # QUESTION FOR HANNA: CONFIRM WE CAN USE THIS NEXT STATE
        # Prepare the observation tensor and ensure it's on the correct device and data type
        one_hot_obs = one_hot_encode_state(next_state.position)

        observation_tensor = T.tensor(one_hot_obs, dtype=T.float32).to(self.learning_agent.critic.device)
        

        # Pass the tensor to the critic and handle possible exceptions
        try:
            val = self.learning_agent.critic(observation_tensor)
            val = T.squeeze(val).item()  # Convert to Python scalar and ensure it's a float
        except Exception as e:
            print("An error occurred during critic evaluation:", e)

        # Now call remember, ensuring all types match and are correctly handled
        try:
            self.learning_agent.remember(observation_tensor.cpu().numpy(), action, val, reward, next_state.terminal)
        except Exception as e:
            print("An error occurred while remembering the state:", e)

        # # Logging simulation data
        # simulation_id = self.simulation_count
        
        # if simulation_id not in self.simulation_logs:
        #     self.simulation_logs[simulation_id] = {
        #         "steps": []
        #     }
        
        # simulation_data = {
        #     "step": depth,
        #     "state": state.position,
        #     "action": action.name,
        #     "reward": reward,
        #     "final_reward": root.z_value ** self._discount_factor
        # }
        # # Append simulation data in reverse order
        # self.simulation_logs[simulation_id]["steps"].insert(0, simulation_data)

        if root[action] is None:
            self._add_qnode(root, action)
                        
        root[action].num_visits += 1
        root[action].r_est = root[action].r_est\
                                + (adj_reward(state, action) - root[action].r_est)\
                                    /root[action].num_visits
        
        if root[action][observation] is None:
            root[action][observation] = RootVNodeRefParticles(self._num_visits_init, \
                                self._z_val_init,
                                self._r_est_init,
                                self._G_init,
                                self._agent.history + ((action, observation),),
                                belief=Particles([]))
                
        root.z_value = root.z_value + (math.exp(root[action].r_est) \
                        * self._simulate(next_state, history + ((action, observation),), \
                        root=root[action][observation], 
                        depth=depth + 1, 
                        exploration_const=exploration_const) - root.z_value) / root.num_visits

        return root.z_value ** self._discount_factor

    def _write_simulation_logs_to_file(self):
        file_path = "simulation_logs.json"  # Or any other file name
        with open(file_path, "w") as f:
            json.dump(self.simulation_logs, f, indent=4)
    
    def _get_u_opt(self, root):
        
        uu_opt_raw = {} # will store the optimal "belief-to-belief" transition probability.
        uu_normaliser = EPSILON
        
        for action in self._agent.all_actions:
            for observation in self._agent.all_observations:
                if root[action] is None or root[action][observation] is None:
                    uu_opt_raw[action, observation] = EPSILON
                else:
                    uu_opt_raw[action, observation] = root[action][observation].num_visits\
                        * root[action][observation].z_value**self._discount_factor\
                        * math.exp(root[action].r_est)\
                            / root.num_visits
                
                uu_normaliser += uu_opt_raw[action, observation]
               
        uu_opt = {ao : uu_weight / uu_normaliser for (ao, uu_weight) in uu_opt_raw.items()}
        
        u_opt = dict()
        u_normaliser = EPSILON
        for action in self._agent.all_actions:
            if root[action] is None or root[action].num_visits == 0:
                u_opt[action] = EPSILON
            else:  
                ha = root[action]
                # print("Action: ", action)
                # print("Num visits: ", ha.num_visits)
                # print("Action reward estimate: ", ha.r_est)
                # u_opt[action] = sum(uu_opt[action, o] for o in ha.children.keys())
                try:
                    pi_a = sum(0.0 if uu_opt[action, observation] == 0 else \
                                ha[observation].num_visits / ha.num_visits \
                                * math.log(ha[observation].num_visits / ha.num_visits \
                                / uu_opt[action, observation]) \
                                for observation in ha.children.keys())
                except:
                    print(ha.num_visits)
                    print([uu_opt[action, observation] for observation in ha.children.keys()])

                u_opt[action] = math.exp(-pi_a)
                                                
            u_normaliser += u_opt[action]
        
        return {a : p / u_normaliser for (a, p) in u_opt.items()}
    
    def _fully_obs_policy_dp(self, state):
        """Computes the fully_observed_policy for a given state and memoises.
        If it has already computed a policy for a given state, this policy is recycled."""
        if self._fully_obs_policy.get(state) is not None:
            return self._fully_obs_policy.get(state)
        
        # TODO: Assumes A Star search here.
        path = self._fully_obs_generator.a_star_search(state.position)
        if path is None or len(path) < 2:
            action = list(self._fully_obs_generator.all_actions)[0]
        else:
            try:
                motion = (path[1][0] - path[0][0], path[1][1] - path[0][1])
            except IndexError:
                print(path)
            action = [a for a in self._fully_obs_generator.all_actions if a.motion == motion][0]
        self._fully_obs_policy[state] = action
        return action
    
    def _VNodeRef(self, agent=None, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNodeRef object."""
        if root:
            # agent cannot be None.
            return RootVNodeRefParticles(self._num_visits_init, 
                                        self._z_val_init,
                                        self._r_est_init,
                                        self._G_init,
                                        agent.history,
                                        belief=copy.deepcopy(agent.belief))
        else:
            if agent is None:
                return VNodeRefParticles(self._num_visits_init, 
                                        self._z_val_init,
                                        self._r_est_init,
                                        self._G_init,
                                        belief=Particles([]))
            else:
                return VNodeRefParticles(self._num_visits_init, 
                                        self._z_val_init,
                                        self._r_est_init,
                                        self._G_init,
                                        belief=copy.deepcopy(agent.belief))