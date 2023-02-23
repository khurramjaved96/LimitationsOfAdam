//
// Created by Khurram Javed on 2022-02-02.
//

#include "../include/learner.h"
#include <iostream>
#include <math.h>
#include <tgmath.h>

LMS::LMS(float step_size, int d) : Learner(step_size, d) {
  bias_step_size = step_size;
  this->counter = 0;
}

Learner::Learner(float step_size, int d) {
  for (int counter = 0; counter < d; counter++) {
    weights.push_back(0);
    step_sizes.push_back(step_size);
    gradients.push_back(0);
  }
  bias_weight = 0;
  bias_gradient = 0;
  dim = d;
}

float LMS::forward(std::vector<float> x) {
  this->counter++;
  float pred = 0;
  for (int counter = 0; counter < dim; counter++) {
    pred += weights[counter] * x[counter];
  }
  //  pred += bias_weight;
  return pred;
}

float Learner::get_dot_product(std::vector<float> my_vec) {
  //  Prod = 1 to take into account the bias term
  float prod = 0;
  for (int c = 0; c < my_vec.size(); c++) {
    prod += my_vec[c] * my_vec[c];
  }
  return prod;
}

void LMS::backward(std::vector<float> x, float pred, float target) {
  float error = target - pred;
  for (int counter = 0; counter < dim; counter++) {
    gradients[counter] -= x[counter] * error;
  }
  bias_gradient -= error;
}

void Learner::zero_grad() {
  for (int c = 0; c < dim; c++) {
    gradients[c] = 0;
  }
  bias_gradient = 0;
}

float Learner::distance_to_target_weights(std::vector<float> target_weights) {
  float avg_distance = 0;
  for (int c = 0; c < dim; c++) {
    avg_distance +=
        (target_weights[c] - weights[c]) * (target_weights[c] - weights[c]);
  }
  return avg_distance / dim;
}

void LMS::update_parameters() {
  for (int c = 0; c < dim; c++) {
    weights[c] -= (step_sizes[c]) * gradients[c];
  }
  //  bias_weight -= (bias_step_size)*bias_gradient;
}

std::vector<float> Learner::get_weights() { return weights; }

RMSPropLMS::RMSPropLMS(float step_size, int d, float beta, float epsilon)
    : LMS(step_size, d) {
  this->beta = beta;
  this->epsilon = epsilon;
  this->v_bias = 0;
  for (int i = 0; i < dim; i++) {
    this->v.push_back(0);
  }
}

void RMSPropLMS::update_RMSProp_statistics() {
  for (int i = 0; i < dim; i++) {
    this->v[i] = this->v[i] * this->beta +
                 (1 - this->beta) * this->gradients[i] * this->gradients[i];
  }
  this->v_bias = this->v_bias * this->beta +
                 (1 - this->beta) * this->bias_gradient * this->bias_gradient;
}

void RMSPropLMS::update_parameters() {
  this->update_RMSProp_statistics();
  for (int i = 0; i < dim; i++)
    this->weights[i] -= this->step_sizes[i] * this->gradients[i] /
                        (sqrt(this->v[i]) + this->epsilon);

  //  this->bias_weight -= this->bias_step_size * bias_gradient
  //  /(sqrt(this->v_bias) + this->epsilon);
}

AdamLMS::AdamLMS(float step_size, int d, float b1, float b2, float epsilon)
    : LMS(step_size, d) {
  this->b1 = b1;
  this->b2 = b2;
  m1_bias = 0;
  m2_bias = 0;
  this->epsilon = epsilon;
  for (int c = 0; c < dim; c++) {
    this->m1.push_back(0);
    this->m2.push_back(0);
  }
}

void AdamLMS::update_adam_statistics() {
  for (int c = 0; c < dim; c++) {
    this->m1[c] = this->m1[c] * this->b1 + (1 - this->b1) * this->gradients[c];
    this->m2[c] = this->m2[c] * this->b2 +
                  (1 - this->b2) * this->gradients[c] * this->gradients[c];
  }
  m1_bias = m1_bias * this->b1 + (1 - this->b1) * bias_gradient;
  m2_bias = m2_bias * this->b2 + (1 - this->b2) * bias_gradient * bias_gradient;
}

void AdamLMS::update_parameters() {
  this->update_adam_statistics();
  for (int c = 0; c < dim; c++) {
    float m1_hat = m1[c] / (1 - pow(b1, this->counter));
    float m2_hat = m2[c] / (1 - pow(b2, this->counter));
    weights[c] -= (step_sizes[c]) * m1_hat / (sqrt(m2_hat) + epsilon);
  }
  float m1_bias_hat = m1_bias / (1 - pow(b1, this->counter));
  float m2_bias_hat = m2_bias / (1 - pow(b2, this->counter));
  //  bias_weight -= (bias_step_size * m1_bias_hat) / (sqrt(m2_bias_hat) +
  //  epsilon);
}

std::vector<float> AdamLMS::get_step_sizes() {
  std::vector<float> effective_step_size;
  for (int c = 0; c < dim; c++) {
    float m1_hat = m1[c] / (1 - pow(b1, this->counter));
    float m2_hat = m2[c] / (1 - pow(b2, this->counter));
    effective_step_size.push_back((step_sizes[c]) / (sqrt(m2_hat) + epsilon));
  }
  return effective_step_size;
}

IDBD::IDBD(float meta_step_size, float step_size, int d) : LMS(step_size, d) {
  for (int c = 0; c < d; c++) {
    this->B.push_back(log(step_size));
    this->step_size_gradients.push_back(0);
    this->h.push_back(0);
    this->step_sizes[c] = exp(this->B[c]);
  }
  h_bias = 0;
  B_bias = log(step_size);
  bias_step_size = exp(B_bias);
  this->meta_step_size = meta_step_size;
}

void IDBD::backward(std::vector<float> x, float pred, float target) {

  float error = target - pred;
  for (int c = 0; c < dim; c++) {
    this->B[c] += meta_step_size * error * x[c] * h[c];
    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - step_sizes[c] * x[c] * x[c]);
    if (temp > 0)
      h[c] = h[c] * temp + step_sizes[c] * error * x[c];
    else
      h[c] = step_sizes[c] * error * x[c];
  }

  B_bias += meta_step_size * error * h_bias;
  bias_step_size = exp(B_bias);
  float temp = (1 - bias_step_size);
  if (temp > 0)
    h_bias = h_bias * temp + bias_step_size * error;
  else
    h_bias = bias_step_size * error;
  LMS::backward(x, pred, target);
}

std::vector<float> IDBD::get_step_sizes() {
  std::vector<float> step_sizes;
  for (int i = 0; i < this->B.size(); i++)
    step_sizes.push_back(exp(this->B[i]));
  return step_sizes;
};

std::vector<float> Learner::get_step_sizes() {
  std::vector<float> step_sizes;
  return this->step_sizes;
}
