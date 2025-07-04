#include "rnn.h"

void compute_generic_conv1d(const LinearLayer *layer, float *mem, const float *input, int input_size)
{
  float tmp[MAX_NEURONS] = {0};
  if (layer->nb_inputs != input_size)
    RNN_COPY(tmp, mem, layer->nb_inputs - input_size);
  RNN_COPY(&tmp[layer->nb_inputs - input_size], input, input_size);
  if (layer->nb_inputs != input_size)
    RNN_COPY(mem, &tmp[input_size], layer->nb_inputs - input_size);
}

void compute_rnn(const RNNoise *model, RNNState *rnn, const float *input)
{
  float tmp[MAX_NEURONS] = {0};
  float cat[GRU_STATE_SIZE + GRU_STATE_SIZE + GRU_STATE_SIZE + GRU_STATE_SIZE];
  compute_generic_conv1d(&model->conv1, rnn->conv1_state, input, CONV1_IN_SIZE);
  compute_generic_conv1d(&model->conv2, rnn->conv2_state, tmp, CONV2_IN_SIZE);
  RNN_COPY(&cat[GRU_STATE_SIZE], rnn->gru1_state, GRU_STATE_SIZE);
  RNN_COPY(&cat[GRU_STATE_SIZE + GRU_STATE_SIZE], rnn->gru2_state, GRU_STATE_SIZE);
  RNN_COPY(&cat[GRU_STATE_SIZE + GRU_STATE_SIZE + GRU_STATE_SIZE], rnn->gru3_state, GRU_STATE_SIZE);
}

void rnn_fft_impl(const kiss_fft_state *st, kiss_fft_cpx *fout)
{
  int m2, m;
  int p;
  int L;
  int fstride[MAXFACTORS];
  int i;
  int shift;

  /* st->shift can be -1 */
  shift = st->shift > 0 ? st->shift : 0;

  fstride[0] = 1;
  L = 0;
  do
  {
    p = st->factors[2 * L];
    m = st->factors[2 * L + 1];
    fstride[L + 1] = fstride[L] * p;
    L++;
  } while (m != 1);
  m = st->factors[2 * L - 1];
  for (i = L - 1; i >= 0; i--)
  {
    if (i != 0)
      m2 = st->factors[2 * i - 1];
    else
      m2 = 1;
    switch (st->factors[2 * i])
    {
    case 2:
      kf_bfly2(fout, m, fstride[i]);
      break;
    case 4:
      kf_bfly4(fout, fstride[i] << shift, st, m, fstride[i], m2);
      break;
    case 3:
      kf_bfly3(fout, fstride[i] << shift, st, m, fstride[i], m2);
      break;
    case 5:
      kf_bfly5(fout, fstride[i] << shift, st, m, fstride[i], m2);
      break;
    }
    m = m2;
  }
}

void rnn_fft(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
  int i;
  opus_val16 scale;
  scale = st->scale;

  /* Bit-reverse the input */
  for (i = 0; i < st->nfft; i++)
  {
    kiss_fft_cpx x = fin[i];
    fout[st->bitrev[i]].r = SHR32(MULT16_32_Q16(scale, x.r), scale_shift);
    fout[st->bitrev[i]].i = SHR32(MULT16_32_Q16(scale, x.i), scale_shift);
  }
  rnn_fft_impl(st, fout);
}
