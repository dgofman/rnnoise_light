#ifndef DENOISE_H
#define DENOISE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "kiss_fft.h"
#include "nnet.h"

extern kiss_fft_state rnn_kfft;
extern const WeightArray *rnnoise_arrays;
extern const float *rnn_dct_table;
extern const float *rnn_half_window;

typedef struct
{
  kiss_fft_cpx delayed_X[FREQ_SIZE];
  kiss_fft_cpx delayed_P[FREQ_SIZE];

  RNNoise model;
  RNNState rnn;
  
  int last_period;
  float last_gain;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
  float analysis_mem[FRAME_SIZE];
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float delayed_Ex[NB_BANDS], delayed_Ep[NB_BANDS], delayed_Exp[NB_BANDS];
} DenoiseState;

/**
 * Denoise a frame of samples
 */
float rnnoise_process_frame(DenoiseState *st, float *out, const float *in);

/**
 * Allocate and initialize a DenoiseState
 */
DenoiseState *rnnoise_create(int full_denoise);

/**
 * Free a DenoiseState produced by rnnoise_create.
 */
void rnnoise_destroy(DenoiseState *st);

#ifdef __cplusplus
}
#endif

#endif /* DENOISE_H */