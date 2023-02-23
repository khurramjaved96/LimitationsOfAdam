//
// Created by Khurram Javed on 2022-02-02.
//

#include "../../include/environments/supervised_tracking.h"
#include <iostream>

Environment::Environment(int seed) : mt(seed){};

WalkingSitting::WalkingSitting(int seed) : Environment(seed), feature_sampling_distribution(0, 1) {
  time_step = 1;
  dimensions = 1;
}

float WalkingSitting::target_function(int t) {
  if (t % 2000 < 1000) {
    return std::abs(10 - (t % 20));
  }
  return 10 + feature_sampling_distribution(this->mt);
}

std::vector<float> WalkingSitting::step() {
  time_step = (time_step + 1) % 20000;
  return get_features();
}

float WalkingSitting::get_target() { return target_function(time_step); }

float WalkingSitting::get_dimension() { return dimensions; }

std::vector<float> WalkingSitting::get_features() {
  return std::vector<float>{1};
}

// Implementation of IDBD environment
IDBDPaper::IDBDPaper(int seed)
    : Environment(seed), feature_sampling_distribution(0, 1),
      weight_change_index_sampler(0, 4) {
  time_step = 0;
  for (int i = 0; i < 5; i++) {
    target_weights.push_back(1);
  }
  for (int i = 0; i < 20; i++) {
    current_x.push_back(0);
  }
  this->generate_x();
}

void IDBDPaper::generate_x() {
  for (int i = 0; i < 20; i++) {
    current_x[i] = feature_sampling_distribution(this->mt);
  }
}

void IDBDPaper::flip_weight() {
  target_weights[weight_change_index_sampler(this->mt)] *= -1;
}

float IDBDPaper::target_function() {
  float target = 0;
  for (int i = 0; i < 5; i++) {
    target += current_x[i] * target_weights[i];
  }
  current_y = target;
  return target;
}

std::vector<float> IDBDPaper::get_target_weights() { return target_weights; }
std::vector<float> IDBDPaper::get_features() { return current_x; }

float IDBDPaper::get_target() { return current_y; }

float IDBDPaper::get_dimension() { return 20; }

std::vector<float> IDBDPaper::step() {
  time_step++;
  generate_x();
  target_function();
  if (time_step % 20 == 0) {
    flip_weight();
  }
  return get_features();
}

