#include "nnet.h"

int FULL_DENOISE = 1;

void compute_rnn(const RNNoise *model, RNNState *rnn, float *gains, float *vad, const float *input)
{
  float tmp[MAX_NEURONS];
  float cat[GRU_STATE_SIZE + GRU_STATE_SIZE + GRU_STATE_SIZE + GRU_STATE_SIZE];
  compute_generic_conv1d(&model->conv1, tmp, rnn->conv1_state, input, CONV1_IN_SIZE, ACTIVATION_TANH);
  compute_generic_conv1d(&model->conv2, cat, rnn->conv2_state, tmp, CONV2_IN_SIZE, ACTIVATION_TANH);
  compute_generic_gru(&model->gru1_input, &model->gru1_recurrent, rnn->gru1_state, cat);
  compute_generic_gru(&model->gru2_input, &model->gru2_recurrent, rnn->gru2_state, rnn->gru1_state);
  compute_generic_gru(&model->gru3_input, &model->gru3_recurrent, rnn->gru3_state, rnn->gru2_state);
  RNN_COPY(&cat[GRU_STATE_SIZE], rnn->gru1_state, GRU_STATE_SIZE);
  RNN_COPY(&cat[GRU_STATE_SIZE + GRU_STATE_SIZE], rnn->gru2_state, GRU_STATE_SIZE);
  RNN_COPY(&cat[GRU_STATE_SIZE + GRU_STATE_SIZE + GRU_STATE_SIZE], rnn->gru3_state, GRU_STATE_SIZE);
  compute_generic_dense(&model->dense_out, gains, cat, ACTIVATION_SIGMOID);
  compute_generic_dense(&model->vad_dense, vad, cat, ACTIVATION_SIGMOID);
}
