#ifndef NNET_H
#define NNET_H

#include "rnn.h"

int linear_init(LinearLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *weights,
  const char *float_weights,
  const char *weights_idx,
  const char *diag,
  const char *scale,
  int nb_inputs,
  int nb_outputs);

void compute_generic_conv1d(const LinearLayer *layer, float *output, float *mem, const float *input, int input_size, int activation);
void compute_generic_gru(const LinearLayer *input_weights, const LinearLayer *recurrent_weights, float *state, const float *in);
void compute_generic_dense(const LinearLayer *layer, float *output, const float *input, int activation);

// rnnoise_data.c
int init_rnnoise(RNNoise *model, const WeightArray *arrays);

#if !defined(__AVX2__)
  #if defined(_MSC_VER)
    #pragma message("AVX2 not enabled. Use /arch:AVX2 or appropriate settings for better performance.")
  #else
    #warning "AVX2 not enabled. Use -march=native or -march=haswell for better performance."
  #endif
#endif

#endif /* NNET_H */
