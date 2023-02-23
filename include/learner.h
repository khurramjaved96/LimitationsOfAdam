//
// Created by Khurram Javed on 2022-02-02.
//

#ifndef INCLUDE_LEARNER_H_
#define INCLUDE_LEARNER_H_
#include <vector>

class Learner {
protected:
  float bias_weight;
  float counter;
  std::vector<float> weights;
  int dim;
  //  float step_size_normalization;
  std::vector<float> step_sizes;
  std::vector<float> gradients;
  float bias_gradient;

public:
  Learner(float step_size, int d);
  float get_dot_product(std::vector<float> my_vec);
  virtual float forward(std::vector<float> x) = 0;
  virtual void backward(std::vector<float> x, float pred, float target) = 0;
  virtual void update_parameters() = 0;
  virtual void zero_grad();
  virtual float distance_to_target_weights(std::vector<float> target_weights);
  std::vector<float> get_weights();
  virtual std::vector<float> get_step_sizes();
};

class LMS : public Learner {
protected:
  float bias_step_size;
  float target_test_mean;

public:
  virtual float forward(std::vector<float> x) override;
  virtual void backward(std::vector<float> x, float pred, float target) override;
  virtual void update_parameters() override;
  LMS(float step_size, int d);
};


class AdamLMS : public LMS {
protected:
  std::vector<float> m1;
  std::vector<float> m2;
  float b1;
  float b2;
  float epsilon;
  float m1_bias;
  float m2_bias;
  void update_adam_statistics();

public:
  std::vector<float> get_step_sizes() override;
  void update_parameters() override;
  AdamLMS(float step_size, int d, float b1, float b2, float epsilon);
};

class IDBD : public LMS {
protected:
  std::vector<float> h;

  std::vector<float> step_size_gradients;
  float h_bias;
  float B_bias;
  float step_size_gradient_bias;
  float meta_step_size;

public:
  std::vector<float> get_step_sizes();
  std::vector<float> B;
  IDBD(float meta_step_size, float step_size, int d);
  virtual void backward(std::vector<float> x, float pred, float target);
};

class RMSPropLMS : public LMS {
protected:
  std::vector<float> v;
  float beta;
  float epsilon;
  float v_bias;
  void update_RMSProp_statistics();

public:
  void update_parameters();
  RMSPropLMS(float step_size, int d, float beta, float epsilon);
};

//
#endif // INCLUDE_LEARNER_H_
