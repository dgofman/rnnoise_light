#include <math.h> // floorf

#include "nnet.h"

// Vector multiply-add for blocks of 8 or 16
#define SGEMV_LOOP(N)                                   \
   for (i = 0; i < rows; i += N)                        \
   {                                                    \
      for (j = 0; j < cols; j++)                        \
      {                                                 \
         const float *w = &weights[j * col_stride + i]; \
         float xj = x[j], *y = &out[i];                 \
         for (int k = 0; k < N; ++k)                    \
            y[k] += w[k] * xj;                          \
      }                                                 \
   }

// compute_generic helper functions
static inline void process_quantized_block(float *y, const opus_int8 *w, const opus_int8 *x, int j)
{
   const int xj[4] = {x[j], x[j + 1], x[j + 2], x[j + 3]};
   if (!FULL_DENOISE)
   {
      for (int r = 0; r < 8; ++r)
         for (int c = 0; c < 4; ++c)
            y[r] += (float)(w[8 * c + r] * xj[c]);
   }
   else
   {
      /* Process each row in the 8-row block */
      y[0] += (float)(w[0] * xj[0] + w[1] * xj[1] + w[2] * xj[2] + w[3] * xj[3]);
      y[1] += (float)(w[4] * xj[0] + w[5] * xj[1] + w[6] * xj[2] + w[7] * xj[3]);
      y[2] += (float)(w[8] * xj[0] + w[9] * xj[1] + w[10] * xj[2] + w[11] * xj[3]);
      y[3] += (float)(w[12] * xj[0] + w[13] * xj[1] + w[14] * xj[2] + w[15] * xj[3]);
      y[4] += (float)(w[16] * xj[0] + w[17] * xj[1] + w[18] * xj[2] + w[19] * xj[3]);
      y[5] += (float)(w[20] * xj[0] + w[21] * xj[1] + w[22] * xj[2] + w[23] * xj[3]);
      y[6] += (float)(w[24] * xj[0] + w[25] * xj[1] + w[26] * xj[2] + w[27] * xj[3]);
      y[7] += (float)(w[28] * xj[0] + w[29] * xj[1] + w[30] * xj[2] + w[31] * xj[3]);
   }
}

// General SGEMV
static inline void sgemv(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   RNN_CLEAR(out, rows);
   if ((rows & 0xf) == 0)
      SGEMV_LOOP(16)
   else if ((rows & 0x7) == 0)
      SGEMV_LOOP(8)
   else
   {
      for (i = 0; i < rows; i++)
      {
         out[i] = 0;
         for (j = 0; j < cols; j++)
            out[i] += weights[j * col_stride + i] * x[j];
      }
   }
}

// Sparse SGEMV 8x4 (unrolled)
static inline void sparse_sgemv8x4(float *out, const float *w, const int *idx, int rows, const float *x)
{
   RNN_CLEAR(out, rows);
   for (int i = 0; i < rows; i += 8)
   {
      int cols = *idx++;
      for (int j = 0; j < cols; j++, w += 32)
      {
         int pos = *idx++;
         float *y = &out[i];
         float xj[4] = {x[pos], x[pos + 1], x[pos + 2], x[pos + 3]};
         for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 4; ++c)
               y[r] += w[8 * c + r] * xj[c];
      }
   }
}

// Compressed sparse GEMV 8x4 (8-bit weights, quantized inputs)
static inline void sparse_cgemv8x4(float *out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
   opus_int8 x[MAX_INPUTS];
   for (int i = 0; i < cols; i++)
      x[i] = (opus_int8)floorf(.5f + 127 * _x[i]);

   for (int i = 0; i < rows; i++)
      out[i] = 0;

   for (int i = 0; i < rows; i += 8)
   {
      int blocks = *idx++;
      for (int j = 0; j < blocks; j++, w += 32)
      {
         int pos = *idx++;
         process_quantized_block(&out[i], w, x, pos);
      }
   }

   for (int i = 0; i < rows; i++)
      out[i] *= scale[i];
}

// Dense CGEMV 8x4 (8-bit weights, quantized inputs)
static inline void cgemv8x4(float *out, const opus_int8 *w, const float *scale, int rows, int cols, const float *_x)
{
   opus_int8 x[MAX_INPUTS];
   for (int i = 0; i < cols; i++)
      x[i] = (opus_int8)floorf(.5f + 127 * _x[i]);

   for (int i = 0; i < rows; i++)
      out[i] = 0;

   for (int i = 0; i < rows; i += 8)
   {
      for (int j = 0; j < cols; j += 4, w += 32)
      {
         process_quantized_block(&out[i], w, x, j);
      }
   }

   for (int i = 0; i < rows; i++)
      out[i] *= scale[i];
}

// Fast tanh approximation
static float tanh_approx(float x)
{
   const float N0 = 952.52801514f, N1 = 96.39235687f, N2 = 0.60863042f;
   const float D0 = 952.72399902f, D1 = 413.36801147f, D2 = 11.88600922f;
   float X2 = x * x;
   float num = fmadd(fmadd(N2, X2, N1), X2, N0);
   float den = fmadd(fmadd(D2, X2, D1), X2, D0);
   return MAX(-1.f, MIN(1.f, num * x / den));
}

// Fast sigmoid approximation
static inline float sigmoid_approx(float x)
{
   return .5f + .5f * tanh_approx(.5f * x);
}

void compute_activation_c(float *output, const float *input, int N, int activation)
{
   if (activation == ACTIVATION_SIGMOID)
   {
      for (int i = 0; i < N; i++)
      {
         output[i] = sigmoid_approx(input[i]);
      }
   }
   else if (activation == ACTIVATION_TANH)
   {
      for (int i = 0; i < N; i++)
      {
         output[i] = tanh_approx(input[i]);
      }
   }
}

void compute_linear_c(const LinearLayer *linear, float *out, const float *in)
{
   int i, M, N;
   const float *bias;
   bias = linear->bias;
   M = linear->nb_inputs;
   N = linear->nb_outputs;
   if (linear->float_weights != NULL)
   {
      if (linear->weights_idx != NULL)
         sparse_sgemv8x4(out, linear->float_weights, linear->weights_idx, N, in);
      else
         sgemv(out, linear->float_weights, N, M, N, in);
   }
   else if (linear->weights != NULL)
   {
      if (linear->weights_idx != NULL)
         sparse_cgemv8x4(out, linear->weights, linear->weights_idx, linear->scale, N, M, in);
      else
         cgemv8x4(out, linear->weights, linear->scale, N, M, in);
   }
   else
   {
      RNN_CLEAR(out, N);
   }
   if (bias != NULL)
   {
      for (i = 0; i < N; i++)
         out[i] += bias[i];
   }
   if (linear->diag)
   {
      for (i = 0; i < M; i++)
      {
         out[i] += linear->diag[i] * in[i];
         out[i + M] += linear->diag[i + M] * in[i];
         out[i + 2 * M] += linear->diag[i + 2 * M] * in[i];
      }
   }
}

void compute_generic_conv1d(const LinearLayer *layer, float *output, float *mem, const float *input, int input_size, int activation)
{
   float tmp[MAX_NEURONS];

   if (layer->nb_inputs != input_size)
      RNN_COPY(tmp, mem, layer->nb_inputs - input_size);
   RNN_COPY(&tmp[layer->nb_inputs - input_size], input, input_size);
   compute_linear_c(layer, output, tmp);
   compute_activation_c(output, output, layer->nb_outputs, activation);
   if (layer->nb_inputs != input_size)
      RNN_COPY(mem, &tmp[input_size], layer->nb_inputs - input_size);
}

void compute_generic_gru(const LinearLayer *input_weights, const LinearLayer *recurrent_weights, float *state, const float *in)
{
   int i;
   int N;
   float zrh[3 * MAX_NEURONS];
   float recur[3 * MAX_NEURONS];
   float *z;
   float *r;
   float *h;

   N = recurrent_weights->nb_inputs;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2 * N];

   compute_linear_c(input_weights, zrh, in);
   compute_linear_c(recurrent_weights, recur, state);
   for (i = 0; i < 2 * N; i++)
      zrh[i] += recur[i];
   compute_activation_c(zrh, zrh, 2 * N, ACTIVATION_SIGMOID);
   for (i = 0; i < N; i++)
      h[i] += recur[2 * N + i] * r[i];
   compute_activation_c(h, h, N, ACTIVATION_TANH);
   for (i = 0; i < N; i++)
      h[i] = z[i] * state[i] + (1 - z[i]) * h[i];
   for (i = 0; i < N; i++)
      state[i] = h[i];
}

void compute_generic_dense(const LinearLayer *layer, float *output, const float *input, int activation)
{
   compute_linear_c(layer, output, input);
   compute_activation_c(output, output, layer->nb_outputs, activation);
}

// linear_init helper functions
static const WeightArray *find_array_entry(const WeightArray *arrays, const char *name)
{
   while (arrays->name && strcmp(arrays->name, name) != 0)
      arrays++;
   return arrays;
}

static const void *find_array_check(const WeightArray *arrays, const char *name, int size)
{
   const WeightArray *a = find_array_entry(arrays, name);
   if (a->name && a->size == size)
      return a->data;
   else
      return NULL;
}

static const float *opt_array_check(const WeightArray *arrays, const char *name, int size, int *error)
{
   const WeightArray *a = find_array_entry(arrays, name);
   *error = (a->name != NULL && a->size != size);
   if (a->name && a->size == size)
      return (float *)a->data;
   else
      return NULL;
}

static const int *find_idx_check(const WeightArray *arrays, const char *name, int nb_in, int nb_out, int *total_blocks)
{
   int remain;
   const WeightArray *a = find_array_entry(arrays, name);
   *total_blocks = 0;
   if (a == NULL)
      return NULL;
   const int *idx_ = (const int *)a->data;
   const int *idx = idx_;
   remain = a->size / SIZEOF(int);
   while (remain > 0)
   {
      int nb_blocks;
      int i;
      nb_blocks = *idx++;
      if (remain < nb_blocks + 1)
         return NULL;
      for (i = 0; i < nb_blocks; i++)
      {
         int pos = *idx++;
         if (pos + 3 >= nb_in || (pos & 0x3))
            return NULL;
      }
      nb_out -= 8;
      remain -= nb_blocks + 1;
      *total_blocks += nb_blocks;
   }
   if (nb_out != 0)
      return NULL;
   return idx_;
}

int linear_init(LinearLayer *layer, const WeightArray *arrays,
                const char *bias,
                const char *weights,
                const char *float_weights,
                const char *weights_idx,
                const char *diag,
                const char *scale,
                int nb_inputs,
                int nb_outputs)
{
   int err;
   layer->bias = NULL;
   layer->weights = NULL;
   layer->float_weights = NULL;
   layer->weights_idx = NULL;
   layer->diag = NULL;
   layer->scale = NULL;
   if (bias != NULL)
   {
      if ((layer->bias = (float *)find_array_check(arrays, bias, nb_outputs * SIZEOF(layer->bias[0]))) == NULL)
         return 1;
   }
   if (weights_idx != NULL)
   {
      int total_blocks;
      if ((layer->weights_idx = find_idx_check(arrays, weights_idx, nb_inputs, nb_outputs, &total_blocks)) == NULL)
         return 1;
      if (weights != NULL)
      {
         if ((layer->weights = (opus_int8 *)find_array_check(arrays, weights, NB_BANDS * total_blocks * SIZEOF(layer->weights[0]))) == NULL)
            return 1;
      }
      if (float_weights != NULL)
      {
         layer->float_weights = opt_array_check(arrays, float_weights, NB_BANDS * total_blocks * SIZEOF(layer->float_weights[0]), &err);
         if (err)
            return 1;
      }
   }
   else
   {
      if (weights != NULL)
      {
         if ((layer->weights = (opus_int8 *)find_array_check(arrays, weights, nb_inputs * nb_outputs * SIZEOF(layer->weights[0]))) == NULL)
            return 1;
      }
      if (float_weights != NULL)
      {
         layer->float_weights = opt_array_check(arrays, float_weights, nb_inputs * nb_outputs * SIZEOF(layer->float_weights[0]), &err);
         if (err)
            return 1;
      }
   }
   if (diag != NULL)
   {
      if ((layer->diag = (float *)find_array_check(arrays, diag, nb_outputs * SIZEOF(layer->diag[0]))) == NULL)
         return 1;
   }
   if (weights != NULL)
   {
      if ((layer->scale = (float *)find_array_check(arrays, scale, nb_outputs * SIZEOF(layer->scale[0]))) == NULL)
         return 1;
   }
   layer->nb_inputs = nb_inputs;
   layer->nb_outputs = nb_outputs;
   return 0;
}