//
// Created by Khurram Javed on 2022-02-02.
//

#ifndef INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_
#define INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_
#include <random>
#include <vector>

class Environment {
protected:
  std::mt19937 mt;

public:
  explicit Environment(int seed);
  virtual std::vector<float> step() = 0;
  virtual std::vector<float> get_features() = 0;
  virtual float get_dimension() = 0;
  virtual float get_target() = 0;
};

class WalkingSitting : public Environment {
  int time_step;
  int dimensions;
  float target_function(int t);
  std::normal_distribution<float> feature_sampling_distribution;

public:
  std::vector<float> get_target_weights();
  std::vector<float> step() override;
  std::vector<float> get_features() override;
  float get_dimension() override;
  float get_target() override;
  explicit WalkingSitting(int seed);
};

class IDBDPaper : public Environment {

  int time_step;
  std::normal_distribution<float> feature_sampling_distribution;
  std::uniform_int_distribution<int> weight_change_index_sampler;
  std::vector<float> target_weights;
  void generate_x();
  std::vector<float> current_x;
  float current_y;
  float target_function();
  void flip_weight();

public:
  std::vector<float> get_target_weights();
  std::vector<float> step() override;
  std::vector<float> get_features() override;
  float get_dimension() override;
  float get_target() override;
  explicit IDBDPaper(int seed);
};

#endif // INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_
