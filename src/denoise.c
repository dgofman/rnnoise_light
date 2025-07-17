#include <stdlib.h> // malloc, free
#include <math.h>   // round, sqrtf, log10f

#include "denoise.h"
#include "pitch.h"

// ERB-spaced band edges (in units of 50 Hz)
int *eband20ms;

// Generate frequency band edges spaced according to the ERB scale (Equivalent Rectangular Bandwidth).
// The result is 34 usable bands (for 32 bands + 2 endpoints), covering ~0 Hz to 20 kHz.
// Units are in multiples of 50 Hz (i.e., 1 = 50 Hz, 20 = 1 kHz, 400 = 20 kHz).
int *generate_eband20ms()
{
  const size_t nb_edges = NB_BANDS + 2; // Start with 35 band edges (will later compress to 34)
  int *asc = (int *)malloc(nb_edges * sizeof(int));
  if (!asc)
    return NULL;

  // Step 1: Generate descending ERB-spaced band edges starting from 20 kHz (400 * 50 Hz)
  int v = 400; // 400 * 50 Hz = 20,000 Hz
  for (size_t i = 0; i < nb_edges; i++)
  {
    double f_hz = v * 50.0;
    double erb = 24.7 * (4.37 * f_hz / 1000.0 + 1.0); // Glasberg & Moore ERB formula
    int step = (int)round(erb / 50.0);                // Convert ERB width back to 50 Hz units
    if (step < 2)
      step = 2; // Enforce minimum step size (100 Hz)
    v -= step;
    size_t index = nb_edges - 1 - i;
    // Step 2: Merge 14 (700 Hz) and 16 (800 Hz) into 15 (750 Hz)
    if (index >= 8)
    {
      asc[index] = v + step; // Shift all elements left to remove 800 Hz (index 8)
    }
    else
    {
      asc[index] = v; // Fill output array in reverse (ascending frequency)
    }
  }

  return asc;
}

DenoiseState *rnnoise_create(int full_denoise)
{
  FULL_DENOISE = full_denoise;
  init_rnnnoise_tables();
  eband20ms = generate_eband20ms();
  DenoiseState *st = (DenoiseState *)malloc(sizeof(DenoiseState));
  if (!st) return NULL;
  memset(st, 0, sizeof(DenoiseState));  // Zero-initialize state
  int ret = init_rnnoise(&st->model, rnnoise_arrays);
  if (ret != 0)
  {
    free(st);
    return NULL;
  }
  return st;
}

static void inverse_transform(float *out, const kiss_fft_cpx *in)
{
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  for (i = 0; i < FREQ_SIZE; i++)
  {
    x[i] = in[i];
  }
  for (; i < WINDOW_SIZE; i++)
  {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  rnn_fft_c(&rnn_kfft, x, y);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE * y[0].r;
  for (i = 1; i < WINDOW_SIZE; i++)
  {
    out[i] = WINDOW_SIZE * y[WINDOW_SIZE - i].r;
  }
}

static void apply_window(float *x)
{
  int i;
  for (i = 0; i < FRAME_SIZE; i++)
  {
    x[i] *= rnn_half_window[i];
    x[WINDOW_SIZE - 1 - i] *= rnn_half_window[i];
  }
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y)
{
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i = 0; i < FRAME_SIZE; i++)
    out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

static void interp_band_gain(float *g, const float *bandE)
{
  int i, j;
  memset(g, 0, FREQ_SIZE);
  for (i = 1; i < NB_BANDS; i++)
  {
    int band_size;
    band_size = eband20ms[i + 1] - eband20ms[i];
    for (j = 0; j < band_size; j++)
    {
      float frac = (float)j / (float)band_size;
      g[eband20ms[i] + j] = (1 - frac) * bandE[i - 1] + frac * bandE[i];
    }
  }
  for (j = 0; j < eband20ms[1]; j++)
    g[j] = bandE[0];
  for (j = eband20ms[NB_BANDS]; j < eband20ms[NB_BANDS + 1]; j++)
    g[j] = bandE[NB_BANDS - 1];
}

static void compute_band_energy(float *bandE, const kiss_fft_cpx *X)
{
  int i;
  float sum[NB_BANDS + 2] = {0};
  for (i = 0; i < NB_BANDS + 1; i++)
  {
    int j;
    int band_size;
    band_size = eband20ms[i + 1] - eband20ms[i];
    for (j = 0; j < band_size; j++)
    {
      float tmp;
      float frac = (float)j / (float)band_size;
      tmp = SQUARE(X[eband20ms[i] + j].r);
      tmp += SQUARE(X[eband20ms[i] + j].i);
      sum[i] += (1 - frac) * tmp;
      sum[i + 1] += frac * tmp;
    }
  }
  sum[1] = (sum[0] + sum[1]) * 2 / 3;
  sum[NB_BANDS] = (sum[NB_BANDS] + sum[NB_BANDS + 1]) * 2 / 3;
  for (i = 0; i < NB_BANDS; i++)
  {
    bandE[i] = sum[i + 1];
  }
}

void rnn_pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P,
                      const float *Ex, const float *Ep, const float *Exp, const float *g)
{
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  float newE[NB_BANDS];
  float norm[NB_BANDS];
  float normf[FREQ_SIZE] = {0};
  for (i = 0; i < NB_BANDS; i++)
  {
    if (Exp[i] > g[i])
      r[i] = 1;
    else
      r[i] = SQUARE(Exp[i]) * (1 - SQUARE(g[i])) / (.001f + SQUARE(g[i]) * (1 - SQUARE(Exp[i])));
    r[i] = sqrtf(MIN(1, MAX(0, r[i])));
    r[i] *= sqrtf(Ex[i] / (1e-8f + Ep[i]));
  }
  interp_band_gain(rf, r);
  for (i = 0; i < FREQ_SIZE; i++)
  {
    X[i].r += rf[i] * P[i].r;
    X[i].i += rf[i] * P[i].i;
  }
  compute_band_energy(newE, X);
  for (i = 0; i < NB_BANDS; i++)
  {
    norm[i] = sqrtf(Ex[i] / (1e-8f + newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i = 0; i < FREQ_SIZE; i++)
  {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
}

static void dct(float *out, const float *in)
{
  int i;
  for (i = 0; i < NB_BANDS; i++)
  {
    int j;
    float sum = 0;
    for (j = 0; j < NB_BANDS; j++)
    {
      sum += in[j] * rnn_dct_table[j * NB_BANDS + i];
    }
    out[i] = sum * sqrtf(2.f / 22.f);
  }
}

static void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P)
{
  int i;
  float sum[NB_BANDS + 2] = {0};
  for (i = 0; i < NB_BANDS + 1; i++)
  {
    int j;
    int band_size;
    band_size = eband20ms[i + 1] - eband20ms[i];
    for (j = 0; j < band_size; j++)
    {
      float tmp;
      float frac = (float)j / (float)band_size;
      tmp = X[eband20ms[i] + j].r * P[eband20ms[i] + j].r;
      tmp += X[eband20ms[i] + j].i * P[eband20ms[i] + j].i;
      sum[i] += (1 - frac) * tmp;
      sum[i + 1] += frac * tmp;
    }
  }
  sum[1] = (sum[0] + sum[1]) * 2 / 3;
  sum[NB_BANDS] = (sum[NB_BANDS] + sum[NB_BANDS + 1]) * 2 / 3;
  for (i = 0; i < NB_BANDS; i++)
  {
    bandE[i] = sum[i + 1];
  }
}

static void forward_transform(kiss_fft_cpx *out, const float *in)
{
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  for (i = 0; i < WINDOW_SIZE; i++)
  {
    x[i].r = in[i];
    x[i].i = 0;
  }
  rnn_fft_c(&rnn_kfft, x, y);
  for (i = 0; i < FREQ_SIZE; i++)
  {
    out[i] = y[i];
  }
}

void rnn_frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in)
{
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i = 0; i < FRAME_SIZE; i++)
    x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(X, x);
  compute_band_energy(Ex, X);
}

void rnn_compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                float *Ex, float *Ep, float *Exp, float *features, const float *in)
{
  int i;
  float E = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE >> 1];
  int pitch_index;
  float gain;
  float *(pre[1]);
  float follow, logMax;
  rnn_frame_analysis(st, X, Ex, in);
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE - FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE], in, FRAME_SIZE);
  pre[0] = &st->pitch_buf[0];
  rnn_pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  rnn_pitch_search(pitch_buf + (PITCH_MAX_PERIOD >> 1), pitch_buf, PITCH_FRAME_SIZE,
                   PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD, &pitch_index);
  pitch_index = PITCH_MAX_PERIOD - pitch_index;

  gain = rnn_remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
                             PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  for (i = 0; i < WINDOW_SIZE; i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index + i];
  apply_window(p);
  forward_transform(P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i = 0; i < NB_BANDS; i++)
    Exp[i] = Exp[i] / sqrtf(.001f + Ex[i] * Ep[i]);
  dct(&features[NB_BANDS], Exp);
  features[2 * NB_BANDS] = .01f * (float)(pitch_index - 300);
  logMax = -2;
  follow = -2;
  for (i = 0; i < NB_BANDS; i++)
  {
    Ly[i] = log10f(1e-2f + Ex[i]);
    Ly[i] = MAX(logMax - 7, MAX(follow - 1.5f, Ly[i]));
    logMax = MAX(logMax, Ly[i]);
    follow = MAX(follow - 1.5f, Ly[i]);
    E += Ex[i];
  }
  if (E < 0.04)
  {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return;
  }
  dct(features, Ly);
  features[0] -= 12;
  features[1] -= 4;
}

void rnn_biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N)
{
  int i;
  for (i = 0; i < N; i++)
  {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0] * xi - a[0] * yi);
    mem[1] = (b[1] * xi - a[1] * yi);
    y[i] = yi;
  }
}

float rnnoise_process_frame(DenoiseState *st, float *out, const float *in)
{
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[FREQ_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE] = {1};
  float vad_prob = 0; //  (Voice Activity Detection)
  static const float a_hp[2] = {-1.99599f, 0.99600f};
  static const float b_hp[2] = {-2, 1};
  rnn_biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  rnn_compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

  compute_rnn(&st->model, &st->rnn, g, &vad_prob, features);
  rnn_pitch_filter(st->delayed_X, st->delayed_P, st->delayed_Ex, st->delayed_Ep, st->delayed_Exp, g);
  for (i = 0; i < NB_BANDS; i++)
  {
    float alpha = .6f;
    /* Cap the decay at 0.6 per frame, corresponding to an RT60 of 135 ms.
       That avoids unnaturally quick attenuation. */
    g[i] = MAX(g[i], alpha * st->lastg[i]);
    /* Compensate for energy change across frame when computing the threshold gain.
       Avoids leaking noise when energy increases (e.g. transient noise). */
    st->lastg[i] = MIN(1.f, g[i] * (st->delayed_Ex[i] + 1e-3f) / (Ex[i] + 1e-3f));
  }
  interp_band_gain(gf, g);
  for (i = 0; i < FREQ_SIZE; i++)
  {
    st->delayed_X[i].r *= gf[i];
    st->delayed_X[i].i *= gf[i];
  }
  frame_synthesis(st, out, st->delayed_X);

  RNN_COPY(st->delayed_X, X, FREQ_SIZE);
  RNN_COPY(st->delayed_P, P, FREQ_SIZE);
  RNN_COPY(st->delayed_Ex, Ex, NB_BANDS);
  RNN_COPY(st->delayed_Ep, Ep, NB_BANDS);
  RNN_COPY(st->delayed_Exp, Exp, NB_BANDS);
  return vad_prob;
}

void rnnoise_destroy(DenoiseState *st)
{
  free_rnnnoise_tables();
  if (eband20ms)
  {
    free(eband20ms);
    eband20ms = NULL;
  }
  free(st);
}