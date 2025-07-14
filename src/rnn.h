#ifndef RNN_H
#define RNN_H

#include <string.h> // memcpy, memmove, memset
#include <stdint.h> // int16_t, int32_t, uint32_t

extern int FULL_DENOISE;

typedef int16_t opus_int16;
typedef int32_t opus_int32;
typedef uint32_t opus_uint32;
typedef signed char opus_int8;

typedef float float_t;

#define MAXFACTORS 8
#define Q15ONE 1.0f
#define EPSILON 1e-15f
#define FRAME_SIZE 480
#define WINDOW_SIZE (2 * FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define NB_BANDS 32
#define NB_FEATURES (2 * NB_BANDS + 1)
#define MAX_NEURONS 1024

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD + PITCH_FRAME_SIZE)

#define M_PIF 3.1415927f

#define CONV1_IN_SIZE 65
#define CONV1_STATE_SIZE (CONV1_IN_SIZE * (2))

#define GRU_STATE_SIZE 384
#define CONV2_IN_SIZE 128
#define CONV2_STATE_SIZE (CONV2_IN_SIZE * (2))

#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH 2

#define WEIGHT_TYPE_float 0
#define WEIGHT_TYPE_int 1
#define WEIGHT_TYPE_int8 3

// nnet.c
#define MAX_INPUTS (2048)
#define fmadd(a, b, c) ((a) * (b) + (c))

#define VAL(name, x, value) (x)
#define SQUARE(x) ((x) * (x))
#define HALF(x) (.5f * (x))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ADD(a, b) ((a) + (b))
#define SUB(a, b) ((a) - (b))
#define MULT(a, b) ((a) * (b))
#define MAC(c, a, b) ((c) + (a) * (b))
#define NEG(x) (-(x))
#define SIZEOF(x) (int)sizeof(x)

#define RNN_COPY(dst, src, n) ((n) > 0 ? memcpy((dst), (src), (size_t)(n) * sizeof(*(dst))) : (void *)0)
#define RNN_MOVE(dst, src, n) (memmove((dst), (src), (n) * (int)sizeof(*(dst)) + 0 * ((dst) - (src))))
#define RNN_CLEAR(dst, n) (memset((dst), 0, (size_t)(n) * sizeof(*(dst))))

typedef struct
{
  const char *name;
  int type;
  int size;
  const void *data;
} WeightArray;

typedef struct
{
  float conv1_state[CONV1_STATE_SIZE];
  float conv2_state[CONV2_STATE_SIZE];
  float gru1_state[GRU_STATE_SIZE];
  float gru2_state[GRU_STATE_SIZE];
  float gru3_state[GRU_STATE_SIZE];
} RNNState;

/* Generic sparse affine transformation. */
typedef struct
{
  const float *bias;
  const opus_int8 *weights;
  const float *float_weights;
  const int *weights_idx;
  const float *diag;
  const float *scale;
  int nb_inputs;
  int nb_outputs;
} LinearLayer;

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

void compute_rnn(const RNNoise *model, RNNState *rnn, float *gains, float *vad, const float *input);

#endif /* RNN_H */
