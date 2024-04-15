import argparse
import copy
import random
from datetime import datetime
import pomdp_py

from grid_map import GridMap
from domain import State
from problem import GridWorldProblem
from utils import init_particles_belief, benchmark_planner


def main(trials_count):
    scale_param = 3
    # ***** BENCHMARK PARAMETERS *****
    simulations = 3000
    planning_time = 60
    trials = trials_count
    nsteps = 40 * scale_param
    discount_factor = 0.99
    num_new_obstacles = 10  # number of new obstacles.
    ref_solver_expl_const = 0.5
    pomcp_expl_const = 300
    # ********************************

    # ***** SMALL GRID MAP *****
    n, m = 20, 20
    k = int(n * m / 15)
    l = int(n * m / 10)
    g = 2

    _grid_map = GridMap.generate_instance(n, m, k, l, 0, g)

    _grid_map_plus = copy.deepcopy(_grid_map)
    not_free_positions = _grid_map.obstacles + _grid_map.goals \
                         + _grid_map.danger_zones + _grid_map.landmarks
    for i in range(num_new_obstacles):
        new_obstacle = GridMap.random_free_position(_grid_map.n, _grid_map.m, not_free_positions)
        _grid_map_plus.obstacles.append(new_obstacle)
        _grid_map_plus.state_positions.remove(new_obstacle)
        not_free_positions.append(new_obstacle)

    # ***** ENLARGE GRID MAP (IF DESIRED) *****
    grid_map = _grid_map.scale(scale_param, scale_param)
    grid_map_plus = _grid_map_plus.scale(scale_param, scale_param)
    # ********************************

    init_state_candidates = [s for s in grid_map_plus.state_positions \
                             if not grid_map_plus.at_goal(s) and s[0] == 1]
    init_states = random.sample(init_state_candidates, k=2)
    init_belief = init_particles_belief(grid_map_plus, init_states=init_states, num_particles=simulations)

    init_pos = random.sample(init_states, 1)[0]
    init_state = State(init_pos,
                       init_pos in grid_map_plus.goals,
                       init_pos in grid_map_plus.landmarks,
                       init_pos in grid_map_plus.danger_zones)

    gridworld = GridWorldProblem(init_state, init_belief, grid_map, scale_param=scale_param)
    gridworld_plus = GridWorldProblem(init_state, init_belief, grid_map_plus, scale_param=scale_param)

    r_max = 20 * scale_param
    r_min = -10 * scale_param
    R_max = 200 * scale_param
    R_min = -100 * scale_param

    rew_scale = (r_min - r_max) / (R_min - R_max)
    rew_shift = r_max / rew_scale - R_max

    print("\n\n***** PROBLEM DEFINITION *****\n")
    print("Gridworld size (n x m):\t\t", gridworld.grid_map.n, "x", gridworld.grid_map.m)
    print("True initial state:\t\t\t", gridworld.init_state.position)
    print("Agent's initial belief:", gridworld.init_belief.get_histogram())
    print("Goal state(s):\t\t\t\t", gridworld.grid_map.goals)
    print("Rmax (goal state):\t\t\t\t", R_max)

    print("\nOriginal environment:")
    gridworld.print_state()

    print("\nDeformed environment:")
    gridworld_plus.print_state()

    print("\n\n***** GENERATING FULLY OBSERVED POLICY GENERATOR ON ORIGINAL ENVIRONMENT *****")

    # ******* A Star *******
    a_star = pomdp_py.AStar(gridworld)

    print("\n\n***** GENERATING FULLY OBSERVED POLICY ON ORIGINAL ENVIRONMENT *****\n")

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

    ref_solver = pomdp_py.RefSolver(max_depth=int(grid_map_plus.n / 2),
                                    max_rollout_depth=grid_map_plus.n,
                                    planning_time=planning_time,
                                    # num_sims=simulations,
                                    fully_obs_policy=a_star_policy,
                                    # fully_obs_generator=a_star,
                                    rew_shift=rew_shift,
                                    rew_scale=rew_scale,
                                    exploration_const=ref_solver_expl_const,
                                    discount_factor=discount_factor)

    # # uniform rollout
    # pomcp_uniform = pomdp_py.POMCP(max_depth=grid_map_plus.n,
    #                               planning_time=planning_time,
    #                               # num_sims=simulations,
    #                               discount_factor=discount_factor,
    #                               exploration_const=pomcp_expl_const,
    #                               rollout_policy=gridworld_plus.agent.policy_model)

    # A* rollout
    pomcp_a_star = pomdp_py.POMCP(max_depth=grid_map_plus.n,
                                  planning_time=planning_time,
                                  # num_sims=simulations,
                                  discount_factor=discount_factor,
                                  exploration_const=pomcp_expl_const,
                                  rollout_policy=a_star_getter)

    # print("\n\n***** RUNNING ONLINE PLANNER(S) *****\n")

    # cr, crd = test_planner(gridworld_plus, ref_solver, nsteps=nsteps,
    #                         discount=discount_factor,
    #                         evolve_step=evolve_step,
    #                         num_new_obstacles=num_new_obstacles)

    # cr, crd = test_planner(gridworld, pomcp_a_star, nsteps=nsteps,
    #                         discount=discount_factor,
    #                         evolve_step=evolve_step,
    #                         num_new_obstacles=num_new_obstacles)

    print("\n\n***** RUNNING BENCHMARKS *****\n")

    print("\nPARAMETERS:\n")
    print("Simulations:", simulations)
    print("Planning time:", planning_time)
    print("Trials:", trials)
    print("Execution steps:", nsteps)
    print("Discount factor:", discount_factor)
    print("New obstacle count:", num_new_obstacles)

    print("\nRefSolver:\n----------------------------")
    print("Reward shift:", ref_solver._rew_shift)
    print("Reward scale:", ref_solver._rew_scale)
    print("Exploration constant:", ref_solver._exploration_const)
    print("Max tree depth:", ref_solver._max_depth)
    print("Max rollout depth:", ref_solver._max_rollout_depth)
    print("----------------------------")

    # print("\nPOMCP (Uniform rollout):\n----------------------------")
    # print("Exploration constant:", pomcp_uniform._exploration_const)
    # print("Max depth:", pomcp_uniform._max_depth)
    # print("----------------------------")

    print("\nPOMCP (A* rollout):\n----------------------------")
    print("Exploration constant:", pomcp_a_star._exploration_const)
    print("Max depth:", pomcp_a_star._max_depth)
    print("----------------------------")

    # ********************************
    results_1 = benchmark_planner(gridworld_plus, ref_solver,
                                  trials=trials,
                                  nsteps=nsteps,
                                  discount_factor=discount_factor)

    # results_2 = benchmark_planner(gridworld_plus, pomcp_uniform,
    #                               trials=trials,
    #                               nsteps=nsteps,
    #                               discount_factor=discount_factor)

    results_3 = benchmark_planner(gridworld_plus, pomcp_a_star,
                                  trials=trials,
                                  nsteps=nsteps,
                                  discount_factor=discount_factor)

    print("\n\n***** RESULTS *****\n")

    print("\nResults RefSolver:")
    for i, v in results_1.items():
        print(i + ":", v)

    # print("\nResults POMCP (unif rollout):")
    # for i, v in results_2.items():
    #     print(i + ":", v)

    print("\nResults POMCP (A* rollout):")
    for i, v in results_3.items():
        print(i + ":", v)

    print("\n\nPreprocessing time fully observed policy:", stop - start, "\n")

    # gridworld.visualise_policy(ref_solver._fully_obs_policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="100", help="Random seed")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")

    args = parser.parse_args()
    print(f"Arguments: {args}")

    random.seed(args.seed)

    main(args.trials)
