#ifndef KISS_FFT_H
#define KISS_FFT_H

#include <stdint.h> // int16_t, int32_t
#include "arch.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MAXFACTORS 8

#define HALF_OF(x) ((x) * .5f)

#define S_MUL(a, b) ((a) * (b))

#define C_MUL(m, a, b)                       \
   do                                        \
   {                                         \
      (m).r = (a).r * (b).r - (a).i * (b).i; \
      (m).i = (a).r * (b).i + (a).i * (b).r; \
   } while (0)

#define C_MULBYSCALAR(c, s) \
   do                       \
   {                        \
      (c).r *= (s);         \
      (c).i *= (s);         \
   } while (0)

#define CHECK_OVERFLOW_OP(a, op, b) /* noop */

#define C_ADD(res, a, b)                 \
   do                                    \
   {                                     \
      CHECK_OVERFLOW_OP((a).r, +, (b).r) \
      CHECK_OVERFLOW_OP((a).i, +, (b).i) \
      (res).r = (a).r + (b).r;           \
      (res).i = (a).i + (b).i;           \
   } while (0)

#define C_SUB(res, a, b)                 \
   do                                    \
   {                                     \
      CHECK_OVERFLOW_OP((a).r, -, (b).r) \
      CHECK_OVERFLOW_OP((a).i, -, (b).i) \
      (res).r = (a).r - (b).r;           \
      (res).i = (a).i - (b).i;           \
   } while (0)

#define C_ADDTO(res, a)                    \
   do                                      \
   {                                       \
      CHECK_OVERFLOW_OP((res).r, +, (a).r) \
      CHECK_OVERFLOW_OP((res).i, +, (a).i) \
      (res).r += (a).r;                    \
      (res).i += (a).i;                    \
   } while (0)

   typedef struct
   {
      double r;
      double i;
   } kiss_fft_cpx;

   typedef struct
   {
      double r;
      double i;
   } kiss_twiddle_cpx;

   typedef struct
   {
      int is_supported;
      void *priv;
   } arch_fft_state;

   typedef struct
   {
      int nfft;
      opus_val16 scale;
      int shift;
      int16_t factors[2 * MAXFACTORS];
      const int32_t *bitrev;
      const kiss_twiddle_cpx *twiddles;
      arch_fft_state *arch_fft;
   } kiss_fft_state;

   void kf_bfly2(kiss_fft_cpx *Fout, int m, int N);
   void kf_bfly3(kiss_fft_cpx *Fout, const size_t fstride, const kiss_fft_state *st, int m, int N, int mm);
   void kf_bfly4(kiss_fft_cpx *Fout, const size_t fstride, const kiss_fft_state *st, int m, int N, int mm);
   void kf_bfly5(kiss_fft_cpx *Fout, const size_t fstride, const kiss_fft_state *st, int m, int N, int mm);

#ifdef __cplusplus
}
#endif

#endif
