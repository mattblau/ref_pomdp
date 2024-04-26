# export PYTHONPATH="/Users/matthewblau/Desktop/ref_pomdp:${PYTHONPATH}"
import argparse
import random
from datetime import datetime
import pomdp_py

from grid_map import GridMap
from domain import State
from problem import GridWorldProblem
from utils import init_particles_belief, benchmark_planner

from PPO_Value_Only.ppo_torch import LearningAgent


def main(trials_count):
    scale_param = 3
    # ***** SMALL GRID MAP *****
    n, m = 20, 20
    obstacles = [(2, i) for i in range(1, 4)] \
                + [(2, i) for i in range(6, 13)] \
                + [(2, i) for i in range(15, 19)] \
                + [(8, i) for i in range(1, 4)] \
                + [(10, i) for i in range(6, 13)] \
                + [(10, i) for i in range(15, 19)] \
                + [(14, i) for i in range(1, 7)] \
                + [(14, i) for i in range(11, 21)] \
                + [(j, 3) for j in range(2, 4)] \
                + [(j, 3) for j in range(5, 11)] \
                + [(j, 3) for j in range(12, 15)] \
                + [(j, 8) for j in range(14, 21)] \
                + [(j, 11) for j in range(14, 17)] \
                + [(j, 11) for j in range(19, 21)] \
                + [(j, 12) for j in range(2, 10)] \
                + [(j, 15) for j in range(2, 10)] \
                + [(j, 18) for j in range(2, 4)] \
                + [(j, 18) for j in range(5, 10)] \

    landmarks = [(15 + i, 12 + j) for i in range(3) for j in range(2)] \
                + [(18 + i, 6 + j) for i in range(3) for j in range(2)] \
                + [(12, 8), (3, 19), (1, 17), (1, 20), (10, 19), (13, 20), (10, 13), (11, 15)] \
                + [(14, 10), (18, 10)]

    danger_zones = [(20, 9), (20, 10)] \
                   + [(6 + i, 19) for i in range(3)] \
                   + [(4 + i, 4 + j) for i in range(6) for j in range(2)]

    goals = [(16, 12), (19, 7)]
    grid_map = GridMap(n, m, obstacles, landmarks, danger_zones, goals)

    grid_map = grid_map.scale(scale_param, scale_param)
    grid_map.danger_zones = [pos for pos in grid_map.danger_zones if pos not in \
                             [(12 + i, 14 + j) for i in range(18) for j in range(2)]]
    grid_map.obstacles = grid_map.obstacles \
                         + [(6 + i, 1 + j) for i in range(3) for j in range(2)] \
                         + [(24 + i, 1 + j) for i in range(3) for j in range(2)] \
                         + [(42 + i, 1 + j) for i in range(3) for j in range(2)]

    # ***** BENCHMARK PARAMETERS *****
    simulations = 3000
    planning_time = 20 # 30
    trials = trials_count
    nsteps = 180
    discount_factor = 0.99
    # ********************************

    init_states = [(1 + i, 13 + j) for i in range(2) for j in range(3)] \
                  + [(1 + i, 40 + j) for i in range(2) for j in range(3)]
    init_belief = init_particles_belief(grid_map, init_states=init_states, num_particles=simulations)

    # init_pos = (1,4)
    init_pos = random.sample(init_states, 1)[0]
    # init_pos = (1, 41)
    init_state = State(init_pos,
                       init_pos in grid_map.goals,
                       init_pos in grid_map.landmarks,
                       init_pos in grid_map.danger_zones)

    gridworld = GridWorldProblem(init_state, init_belief, grid_map, scale_param=scale_param)

    r_max = 60
    r_min = -30
    R_max = 600
    R_min = -300

    rew_scale = (r_min - r_max) / (R_min - R_max)
    rew_shift = r_max / rew_scale - R_max

    print("\n\n***** PROBLEM DEFINITION *****\n")
    print("Gridworld size (n x m):\t\t", grid_map.n, "x", grid_map.m)
    print("True initial state:\t\t\t", init_state.position)
    print("Agent's initial belief:", init_belief.get_histogram())
    print("Goal state(s):\t\t\t\t", grid_map.goals)
    print("Rmax (goal state):\t\t\t\t", R_max)
    print("Rmin (danger zone): \t\t\t\t\t", R_min)

    gridworld.print_state()

    print("\n\n***** GENERATING FULLY OBSERVED POLICY GENERATOR *****")

    # ******* A Star *******
    a_star = pomdp_py.AStar(gridworld)

    print("\n\n***** GENERATING FULLY OBSERVED POLICY *****\n")

    start = datetime.now()

    a_star_policy = a_star.a_star_policy(gridworld.agent)

    # class RolloutPolicy(PolicyModel):
    #     def rollout(self, state, history):
    #         """rollout(self, State state, tuple history=None)"""
    #         pass

    class AStarRollout(pomdp_py.RolloutPolicy):
        """A rollout policy that chooses actions using an a* policy."""

        def __init__(self, a_star_policy):
            self._a_star_policy = a_star_policy

        def rollout(self, state, history=None):
            return self._a_star_policy[state]

    a_star_getter = AStarRollout(a_star_policy)

    stop = datetime.now()

    gridworld.visualise_policy(a_star_policy)

    print("Preprocessing time fully observed policy:", stop - start)

    print("\n\n***** BUILDING PLANNER(S) *****\n")

    # Thinking
    # For Cartpole, we learn every 20 steps, with a batch size of 5 over 4 epochs
    # For this environment, we learn every ~90 steps, with a batch size of 5 over 4 epochs
    # We should be learning every ~90 steps, with a batch size of 25 over 4 epochs

    # Generating learning agent
    batch_size = 32 # 64
    n_epochs = 10
    alpha = 0.0003
    n_actions = 4 # North, South, East, West
    input_dims = (3600,)

    agent = LearningAgent(n_actions=n_actions, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=input_dims)

    ref_solver_learn = pomdp_py.RefSolverLearn(learning_agent=agent,
                                    max_depth=90, #90
                                    max_rollout_depth=180,
                                    planning_time=planning_time,
                                    # num_sims=simulations,
                                    fully_obs_policy=a_star_policy,
                                    # fully_obs_generator=a_star,
                                    rew_shift=rew_shift,
                                    rew_scale=rew_scale,
                                    # rew_shift=0,
                                    # rew_scale=1/3,
                                    exploration_const=0.5,
                                    discount_factor=discount_factor)

    ref_solver = pomdp_py.RefSolver(max_depth=120,
                                    max_rollout_depth=180,
                                    planning_time=planning_time,
                                    # num_sims=simulations,
                                    fully_obs_policy=a_star_policy,
                                    # fully_obs_generator=a_star,
                                    rew_shift=rew_shift,
                                    rew_scale=rew_scale,
                                    # rew_shift=0,
                                    # rew_scale=1/3,
                                    exploration_const=0.5,
                                    discount_factor=discount_factor)

    # uniform rollout
    # pomcp = pomdp_py.POMCP(max_depth=grid_map.n * 3,
    #                         planning_time=planning_time,
    #                         # num_sims=simulations,
    #                         discount_factor=0.99,
    #                         exploration_const=1000,
    #                         rollout_policy=gridworld.agent.policy_model)

    # A* rollout
    pomcp_a_star = pomdp_py.POMCP(max_depth=180,
                                  planning_time=planning_time,
                                  # num_sims=simulations,
                                  discount_factor=discount_factor,
                                  exploration_const=300,
                                  rollout_policy=a_star_getter)

    # print("\n\n***** RUNNING ONLINE PLANNER(S) *****\n")

    # cr, crd = test_planner(gridworld, ref_solver, nsteps=nsteps, discount=discount_factor)

    # cr, crd = test_planner(gridworld, pomcp, nsteps=nsteps, discount=discount_factor)

    # cr, crd = test_planner(gridworld, pomcp_a_star, nsteps=nsteps, discount=discount_factor)

    print("\n\n***** RUNNING BENCHMARKS *****\n")

    print("\nPARAMETERS:\n")
    print("Simulations:", simulations)
    print("Planning time:", planning_time)
    print("Trials:", trials)
    print("Planning horizon:", nsteps)
    print("Discount factor:", discount_factor)

    print("\nRefSolverLearn:\n----------------------------")
    print("Reward shift:", ref_solver_learn._rew_shift)
    print("Reward scale:", ref_solver_learn._rew_scale)
    print("Exploration constant:", ref_solver_learn._exploration_const)
    print("Max tree depth:", ref_solver_learn._max_depth)
    print("Max rollout depth:", ref_solver_learn._max_rollout_depth)
    print("----------------------------")

    print("\nRefSolver:\n----------------------------")
    print("Reward shift:", ref_solver._rew_shift)
    print("Reward scale:", ref_solver._rew_scale)
    print("Exploration constant:", ref_solver._exploration_const)
    print("Max tree depth:", ref_solver._max_depth)
    print("Max rollout depth:", ref_solver._max_rollout_depth)
    print("----------------------------")

    # print("\nPOMCP (Uniform rollout):\n----------------------------")
    # print("Exploration constant:", pomcp._exploration_const)
    # print("Max depth:", pomcp._max_depth)
    # print("----------------------------")

    print("\nPOMCP (A* rollout):\n----------------------------")
    print("Exploration constant:", pomcp_a_star._exploration_const)
    print("Max depth:", pomcp_a_star._max_depth)
    print("----------------------------")

    # ********************************

    results_1 = benchmark_planner(gridworld, ref_solver_learn,
                                  trials=trials,
                                  nsteps=nsteps,
                                  discount_factor=discount_factor)
    
    results_2 = benchmark_planner(gridworld, ref_solver,
                                  trials=trials,
                                  nsteps=nsteps,
                                  discount_factor=discount_factor)

    # results_3 = benchmark_planner(gridworld, pomcp_a_star,
    #                               trials=trials,
    #                               nsteps=nsteps,
    #                               discount_factor=discount_factor)

    print("\n\n***** RESULTS *****\n")

    print("\nResults RefSolverLearn:")
    for i, v in results_1.items():
        print(i + ":", v)

    print("\nResults RefSolver:")
    for i, v in results_2.items():
        print(i + ":", v)

    # print("\nResults POMCP (A* rollout):")
    # for i, v in results_3.items():
    #     print(i + ":", v)

    print("\n\nPreprocessing time fully observed policy:", stop - start, "\n")

    gridworld.visualise_policy(ref_solver._fully_obs_policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"), help="Random seed")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials")

    args = parser.parse_args()
    print(f"Arguments: {args}")

    random.seed(args.seed)

    main(args.trials)
