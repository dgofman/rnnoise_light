#ifndef PITCH_H
#define PITCH_H

#include "rnn.h"

void rnn_pitch_downsample(float_t *x[], float_t *x_lp, int len, int C);

void rnn_pitch_search(const float_t *x_lp, float_t *y, int len, int max_pitch, int *pitch);

float_t rnn_remove_doubling(float_t *x, int maxperiod, int minperiod, int N, int *T0, int prev_period, float_t prev_gain);

#endif /* PITCH_H */
