#vase.py
#pseudocode of run_trpo_vase.py

# 1. imports from various places
# 2. define function run_trpo_vase(env,nRuns,seed_base,sigma_c,ablation_mode=False)

# 3. for num_runs=nRuns:
# [4. define the ~~mdp~~ on the basis of env_string=env, n_itr, max_path_length, type.
#     Allows: mountaincar, cartpole, doublependulum, halfcheetah, locomotion, ant, lunarlander
# 5. defining on the basis of 'type' (determined by env): step_size, replay_pool_size, policy_hidden_sizes,
#     unn_n_hidden, unn_layers_type, baseline = ~~GaussianMLPBaseline~~(env_spec=mdp.spec, regressor_args={...})
# 6. define policy = ~~GaussianMLPPolicy~~(env_spec, hidden_sizes, hidden_nonlinearity), define algo = ~~TRPO~~(......)
# 7. define logging directories on basis of ablation mode status
# 8. executes ~~.instrument.run_experiment_lite(...)~~]

# 9. __name__ == "__main__":
# 10. args=parser.ArgumentParser()
# 11. executes run_trpo_vase(args)

# key aspects of ~~mdp~~:
# is the instantiation of a class that (in the case of env='mountaincar') inherits from Box2DEnv and Serializable.
# it is using these '@autoargs' and '@overrides' that I don't understand.
# it defines 4 functions (with '@overrides'):
#     compute_reward(self, action); is_current_done(self); reset(self); action_from_keys(self, keys)

# key aspects of ~~GaussianMLPBaseline~~:
# is a class that inherits from Baseline and Parameterized.
# defines 4 functions (with '@overrides'):
#    fit(self, paths); predict(self, path); get_param_values(self, **tags);
#    set_param_values(self, flattened_params, **tags)

# key aspects of ~~GaussianMLPPolicy~~:
# is a class that inherits from StochasticPolicy and LasagnePowered.
# defines a mean_network and std_network from the imported MLP class in the __init__() along with other attributes.
# defines 6 functions (some with '@overrides'):
#    dist_info_sym(self, obs_var, state_info_vars=None); get_action(self, observation); get_actions(self, observations);
#    get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars); log_diagnostics(self, paths);
#    distribution(self).

# key aspects of ~~TRPO~~:
# is a class that inherits from NPO and Serializable.
# just has an __init__() method takes optimizer=None, optimizer_args=None, **kwargs.
# Does Serializable.quick_init() then ~~NPO~~ is initialized using the optimizer(optimizer_args) and **kwargs.

# key aspects of ~~NPO~~:
# is a class that inherits from ~~BatchPolopt~~.
# defines 3 functions (with '@overrides'): init_opt(self); optimize_policy(self, itr, samples_data);
#   get_itr_snapshot(self, itr, samples_data).

# key aspects of ~~BatchPolopt~~:
# "Base class for batch sampling-based policy optimization methods. This includes various policy gradient methods like:
#    vpg, npg, ppo, trpo, etc. Inherits from RLAlgorithm.
# Has alot of variables including env, policy, baseline, n_itr, start_itr, batch_size, max_path_length, discount, etc.
# Has 9 functions. Most important seems to be ~~train(self)~~.

# key aspects of ~~train(self)~~:
# function goes from lines 203 - 335.
# starts with ~~BNN~~ initialization, where self.bnn = bnn.BNN(...) is executed. sandbox.vase.dynamics.bnn is source.
# defines replay pools and worker.
# for each itr: rollouts are obtained and processed. if using replay pool, fill it, and then train the dynamics model.
# logging losses and rewards, and optional plotting, then worker shutdown.

# key aspects of ~~BNN()~~
# is a class that inherits from LasagnePowered and Serializable, based on Blundell2016.
# in the __init__(), calls self.build_network() --> self.network, and initializes LasagnePowered(self, [self.network]).
# also calls self.build_model() which uses various parts (loss/surprise/etc) to build pred_fn/train_fn/surprise_fn.
# the network is made up of a mix of dense layers and probabilistic layers (for exploration??).

# key aspects of ~~instrument.run_experiment_lite(...)~~:
# has lots 1376 lines of code in this file. run_experiment_lite runs from line 338 to line 605.
# it is a function with a whole ton of args.
# the function "serializes the stubbed method call, e.g. algo.train(), then runs experiment using the specified mode".
# 'serializes' means to turn into a format that can be stored or transmitted over a network and reconstructed later.
# the available modes are: if mode.__call__() exists; "local"; "local_docker"; "ec2"; "lab_kube".
# whatever mode is used, the script run is "~~scripts/run_experiment_lite.py"~~.

# key aspects of ~~scripts/run_experiment_lite.py~~:
# houses a small 100 line function run_experiment(argv), that it runs whenever the file is called.
# From f(argv) it takes arguments from both the command line (from where it is called), and from the argument parser.
# initializes to use a certain number of parallel workers to perform rollouts, each with the same random seed.
# defines plotting and logging (of tabular/text/params/variant data through rllab.misc.logger).
# runs training of the algo, either resuming from loaded job,
#     or from scratch using pickled data (args.args_data: "pickled data for stub objects"), with or without cloudpickle.


