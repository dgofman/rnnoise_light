#ifndef KISS_FFT_H
#define KISS_FFT_H

#include "rnn.h"

typedef struct
{
   float_t r;
   float_t i;
} kiss_fft_cpx;

typedef struct {
   int nfft;
   float_t scale;
   int shift;
   opus_int16 factors[2 * MAXFACTORS];
   opus_int32 *bitrev;
   kiss_fft_cpx *twiddles;
} kiss_fft_state;

void rnn_fft_c(const kiss_fft_state *cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout);

// rnnoise_tables.c
void init_rnnnoise_tables();
void free_rnnnoise_tables();

#endif /* KISS_FFT_H */
