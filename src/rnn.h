#ifndef RNN_H_
#define RNN_H_

#include <string.h> // memcpy, memmove, memset
#include "nnet.h"
#include "kiss_fft.h"

#define MAX_NEURONS 1024

#define GRU_STATE_SIZE 384

#define CONV1_IN_SIZE 65
#define CONV2_IN_SIZE 128

#define CONV1_STATE_SIZE (CONV1_IN_SIZE * (2))
#define CONV2_STATE_SIZE (CONV2_IN_SIZE * (2))

#define RNN_COPY(dst, src, n) (memcpy((dst), (src), (n) * sizeof(*(dst)) + 0 * ((dst) - (src))))
#define RNN_MOVE(dst, src, n) (memmove((dst), (src), (n) * sizeof(*(dst)) + 0 * ((dst) - (src))))
#define RNN_CLEAR(dst, n) (memset((dst), 0, (n) * sizeof(*(dst))))

typedef struct
{
  LinearLayer conv1;
  LinearLayer conv2;
  LinearLayer gru1_input;
  LinearLayer gru1_recurrent;
  LinearLayer gru2_input;
  LinearLayer gru2_recurrent;
  LinearLayer gru3_input;
  LinearLayer gru3_recurrent;
  LinearLayer dense_out;
  LinearLayer vad_dense;
} RNNoise;

typedef struct
{
  float conv1_state[CONV1_STATE_SIZE];
  float conv2_state[CONV2_STATE_SIZE];
  float gru1_state[GRU_STATE_SIZE];
  float gru2_state[GRU_STATE_SIZE];
  float gru3_state[GRU_STATE_SIZE];
} RNNState;

int init_rnnoise(RNNoise *model, const WeightArray *arrays);
void compute_rnn(const RNNoise *model, RNNState *rnn, const float *input);
void rnn_fft(const kiss_fft_state *cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout);

#endif /* RNN_H_ */
