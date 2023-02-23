//
// Created by Khurram Javed on 2022-01-08.
//

#include "include/environments/supervised_tracking.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include "include/learner.h"
#include "include/network_factory.h"
#include "include/utils.h"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  int T = my_experiment->get_int_param("steps");
  int no_of_features = 1;

  //  Initialize database tables
  Metric error_metric =
      Metric(my_experiment->database_name, "error_table",
             std::vector<std::string>{"run", "step", "seed",
                                      "mean_sum_of_errors", "asymptotic_error"},
             std::vector<std::string>{"int", "int", "int", "real", "real"},
             std::vector<std::string>{"run", "step", "seed"});

  Metric alpha_metric = Metric(
      my_experiment->database_name, "alpha_table",
      std::vector<std::string>{"run", "step", "seed", "feature_no", "alpha"},
      std::vector<std::string>{"int", "int", "int", "int", "real"},
      std::vector<std::string>{"run", "step", "seed", "feature_no"});

  Metric target_weights_metric =
      Metric(my_experiment->database_name, "target_weights",
             std::vector<std::string>{"run", "step", "seed", "prediction",
                                      "target"},
             std::vector<std::string>{"int", "int", "int", "real", "real"},
             std::vector<std::string>{"run", "step", "seed"});

  //  Repeat experiment for seed number of times
  for (int seed = 0; seed < my_experiment->get_int_param("seeds"); seed++) {
    //    Initialize the environment

    WalkingSitting *env = new WalkingSitting(seed);

    float sum_of_error = 0;
    float last_20k_steps_error = 0;
    Learner *network = NetworkFactory::get_learner(my_experiment);

    for (int step = 0; step < T; step++) {
      //      Get next sample from the world
      auto x = env->step();
      //      Make a prediction
      float pred = network->forward(x);
      //      Get target/label given by the underlying target function after the
      //      agent has made the prediction
      float target = env->get_target();
      //      Compute the squared error
      float squared_error = (target - pred) * (target - pred);
      //      Set the gradient accumulation vector to zero
      network->zero_grad();
      //      Add gradient of the 1/2 (target - pred)^2 w.r.t the learnable
      //      parameters to the gradient accumulation vector
      network->backward(x, pred, target);
      //      Update the parameters using the stored gradients
      network->update_parameters();
      //      Update the sum of errors so far
      sum_of_error += squared_error;
      //      Update asymptotic error (Error on last 20k steps).
      if (T - step < 20000) {
        last_20k_steps_error += squared_error;
      }
      if(step == T-1){
        std::vector<float> alphas = network->get_step_sizes();
        for (int c = 0; c < no_of_features; c++) {
          std::vector<std::string> cur_alphas;
          cur_alphas.push_back(
              std::to_string(my_experiment->get_int_param("run")));
          cur_alphas.push_back(std::to_string(step));
          cur_alphas.push_back(std::to_string(seed));
          cur_alphas.push_back(std::to_string(c));
          cur_alphas.push_back(std::to_string(alphas[c]));
          alpha_metric.record_value(cur_alphas);
        }


        std::vector<std::string> cur_alphas;
        cur_alphas.push_back(
            std::to_string(my_experiment->get_int_param("run")));
        cur_alphas.push_back(std::to_string(step));
        cur_alphas.push_back(std::to_string(seed));
        cur_alphas.push_back(std::to_string(pred));
        cur_alphas.push_back(std::to_string(target));
        target_weights_metric.record_value(cur_alphas);

      }
    }

    //    Push results in the database
    std::vector<std::string> cur_error;
    cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
    cur_error.push_back(std::to_string(T));
    cur_error.push_back(std::to_string(seed));
    cur_error.push_back(std::to_string(sum_of_error / T));
    cur_error.push_back(std::to_string(last_20k_steps_error/20000));
    error_metric.record_value(cur_error);
  }
  alpha_metric.commit_values();
  error_metric.commit_values();
  target_weights_metric.commit_values();

  std::cout << "Done\n";
}
