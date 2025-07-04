#include <math.h> // abs, sqrt

#include "pitch.h"
#include "rnn.h"

void rnn_lpc(
    opus_val16 *_lpc,     /* out: [0...p-1] LPC coefficients      */
    const opus_val32 *ac, /* in:  [0...p] autocorrelation values  */
    int p)
{
   int i, j;
   opus_val32 r;
   opus_val32 error = ac[0];
   float *lpc = _lpc;

   RNN_CLEAR(lpc, p);
   if (ac[0] != 0)
   {
      for (i = 0; i < p; i++)
      {
         /* Sum up this iteration's reflection coefficient */
         opus_val32 rr = 0;
         for (j = 0; j < i; j++)
            rr += MULT32_32_Q31(lpc[j], ac[i - j]);
         rr += SHR32(ac[i + 1], 3);
         r = -SHL32(rr, 3) / error;
         /*  Update LPC coefficients and total error */
         lpc[i] = SHR32(r, 3);
         for (j = 0; j < (i + 1) >> 1; j++)
         {
            opus_val32 tmp1, tmp2;
            tmp1 = lpc[j];
            tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + MULT32_32_Q31(r, tmp2);
            lpc[i - 1 - j] = tmp2 + MULT32_32_Q31(r, tmp1);
         }

         error = error - MULT32_32_Q31(MULT32_32_Q31(r, r), error);
         /* Bail out once we get 30 dB gain */
         if (error < .001f * ac[0])
            break;
      }
   }
}

int rnn_autocorr(
    const opus_val16 *x, /*  in: [0...n-1] samples x   */
    opus_val32 *ac,      /* out: [0...lag-1] ac values */
    const opus_val16 *window,
    int overlap,
    int lag,
    int n)
{
   opus_val32 d;
   int i, k;
   int fastN = n - lag;
   int shift;
   const opus_val16 *xptr;
   opus_val16 xx[PITCH_BUF_SIZE / 2];
   if (overlap == 0)
   {
      xptr = x;
   }
   else
   {
      for (i = 0; i < n; i++)
         xx[i] = x[i];
      for (i = 0; i < overlap; i++)
      {
         xx[i] = MULT16_16_Q15(x[i], window[i]);
         xx[n - i - 1] = MULT16_16_Q15(x[n - i - 1], window[i]);
      }
      xptr = xx;
   }
   shift = 0;
   rnn_pitch_xcorr(xptr, xptr, ac, fastN, lag + 1);
   for (k = 0; k <= lag; k++)
   {
      for (i = k + fastN, d = 0; i < n; i++)
         d = MAC16_16(d, xptr[i], xptr[i - k]);
      ac[k] += d;
   }
   return shift;
}

static void find_best_pitch(opus_val32 *xcorr, opus_val16 *y, int len,
                            int max_pitch, int *best_pitch)
{
   int i, j;
   opus_val32 Syy = 1;
   opus_val16 best_num[2];
   opus_val32 best_den[2];

   best_num[0] = -1;
   best_num[1] = -1;
   best_den[0] = 0;
   best_den[1] = 0;
   best_pitch[0] = 0;
   best_pitch[1] = 1;
   for (j = 0; j < len; j++)
      Syy = ADD32(Syy, SHR32(MULT16_16(y[j], y[j]), yshift));
   for (i = 0; i < max_pitch; i++)
   {
      if (xcorr[i] > 0)
      {
         opus_val16 num;
         opus_val32 xcorr16;
         xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));
         num = MULT16_16_Q15(xcorr16, xcorr16);
         if (MULT16_32_Q15(num, best_den[1]) > MULT16_32_Q15(best_num[1], Syy))
         {
            if (MULT16_32_Q15(num, best_den[0]) > MULT16_32_Q15(best_num[0], Syy))
            {
               best_num[1] = best_num[0];
               best_den[1] = best_den[0];
               best_pitch[1] = best_pitch[0];
               best_num[0] = num;
               best_den[0] = Syy;
               best_pitch[0] = i;
            }
            else
            {
               best_num[1] = num;
               best_den[1] = Syy;
               best_pitch[1] = i;
            }
         }
      }
      Syy += SHR32(MULT16_16(y[i + len], y[i + len]), yshift) - SHR32(MULT16_16(y[i], y[i]), yshift);
      Syy = MAX32(1, Syy);
   }
}

static void celt_fir5(const opus_val16 *x,
                      const opus_val16 *num,
                      opus_val16 *y,
                      int N,
                      opus_val16 *mem)
{
   int i;
   opus_val16 num0, num1, num2, num3, num4;
   opus_val32 mem0, mem1, mem2, mem3, mem4;
   num0 = num[0];
   num1 = num[1];
   num2 = num[2];
   num3 = num[3];
   num4 = num[4];
   mem0 = mem[0];
   mem1 = mem[1];
   mem2 = mem[2];
   mem3 = mem[3];
   mem4 = mem[4];
   for (i = 0; i < N; i++)
   {
      opus_val32 sum = SHL32(EXTEND32(x[i]), SIG_SHIFT);
      sum = MAC16_16(sum, num0, mem0);
      sum = MAC16_16(sum, num1, mem1);
      sum = MAC16_16(sum, num2, mem2);
      sum = MAC16_16(sum, num3, mem3);
      sum = MAC16_16(sum, num4, mem4);
      mem4 = mem3;
      mem3 = mem2;
      mem2 = mem1;
      mem1 = mem0;
      mem0 = x[i];
      y[i] = ROUND16(sum, SIG_SHIFT);
   }
   mem[0] = mem0;
   mem[1] = mem1;
   mem[2] = mem2;
   mem[3] = mem3;
   mem[4] = mem4;
}

void rnn_pitch_downsample(celt_sig *x[], opus_val16 *x_lp,
                          int len, int C)
{
   int i;
   opus_val32 ac[5];
   opus_val16 tmp = Q15ONE;
   opus_val16 lpc[4], mem[5] = {0, 0, 0, 0, 0};
   opus_val16 lpc2[5];
   opus_val16 c1 = QCONST16(.8f, 15);
   for (i = 1; i < len >> 1; i++)
      x_lp[i] = SHR32(HALF32(HALF32(x[0][(2 * i - 1)] + x[0][(2 * i + 1)]) + x[0][2 * i]), shift);
   x_lp[0] = SHR32(HALF32(HALF32(x[0][1]) + x[0][0]), shift);
   if (C == 2)
   {
      for (i = 1; i < len >> 1; i++)
         x_lp[i] += SHR32(HALF32(HALF32(x[1][(2 * i - 1)] + x[1][(2 * i + 1)]) + x[1][2 * i]), shift);
      x_lp[0] += SHR32(HALF32(HALF32(x[1][1]) + x[1][0]), shift);
   }

   rnn_autocorr(x_lp, ac, NULL, 0,
                4, len >> 1);

   /* Noise floor -40 dB */
   ac[0] *= 1.0001f;

   /* Lag windowing */
   for (i = 1; i <= 4; i++)
   {
      ac[i] -= ac[i] * (.008f * i) * (.008f * i);
   }

   rnn_lpc(lpc, ac, 4);
   for (i = 0; i < 4; i++)
   {
      tmp = MULT16_16_Q15(QCONST16(.9f, 15), tmp);
      lpc[i] = MULT16_16_Q15(lpc[i], tmp);
   }
   /* Add a zero */
   lpc2[0] = lpc[0] + QCONST16(.8f, SIG_SHIFT);
   lpc2[1] = lpc[1] + MULT16_16_Q15(c1, lpc[0]);
   lpc2[2] = lpc[2] + MULT16_16_Q15(c1, lpc[1]);
   lpc2[3] = lpc[3] + MULT16_16_Q15(c1, lpc[2]);
   lpc2[4] = MULT16_16_Q15(c1, lpc[3]);
   celt_fir5(x_lp, lpc2, x_lp, len >> 1, mem);
}

void rnn_pitch_xcorr(const opus_val16 *_x, const opus_val16 *_y,
                     opus_val32 *xcorr, int len, int max_pitch)
{
   int i;
   for (i = 0; i < max_pitch - 3; i += 4)
   {
      opus_val32 sum[4] = {0, 0, 0, 0};
      xcorr_kernel(_x, _y + i, sum, len);
      xcorr[i] = sum[0];
      xcorr[i + 1] = sum[1];
      xcorr[i + 2] = sum[2];
      xcorr[i + 3] = sum[3];
   }
   /* In case max_pitch isn't a multiple of 4, do non-unrolled version. */
   for (; i < max_pitch; i++)
   {
      opus_val32 sum;
      sum = celt_inner_prod(_x, _y + i, len);
      xcorr[i] = sum;
   }
}

void rnn_pitch_search(const opus_val16 *x_lp, opus_val16 *y,
                      int len, int max_pitch, int *pitch)
{
   int i, j;
   int lag;
   int best_pitch[2] = {0, 0};
   int offset;
   opus_val16 x_lp4[PITCH_FRAME_SIZE >> 2];
   opus_val16 y_lp4[(PITCH_FRAME_SIZE + PITCH_MAX_PERIOD) >> 2];
   opus_val32 xcorr[PITCH_MAX_PERIOD >> 1];
   lag = len + max_pitch;

   /* Downsample by 2 again */
   for (j = 0; j < len >> 2; j++)
      x_lp4[j] = x_lp[2 * j];
   for (j = 0; j < lag >> 2; j++)
      y_lp4[j] = y[2 * j];
   /* Coarse search with 4x decimation */
   rnn_pitch_xcorr(x_lp4, y_lp4, xcorr, len >> 2, max_pitch >> 2);

   find_best_pitch(xcorr, y_lp4, len >> 2, max_pitch >> 2, best_pitch);

   /* Finer search with 2x decimation */
   for (i = 0; i < max_pitch >> 1; i++)
   {
      opus_val32 sum;
      xcorr[i] = 0;
      if (abs(i - 2 * best_pitch[0]) > 2 && abs(i - 2 * best_pitch[1]) > 2)
         continue;
      sum = celt_inner_prod(x_lp, y + i, len >> 1);
      xcorr[i] = MAX32(-1, sum);
   }
   find_best_pitch(xcorr, y, len >> 1, max_pitch >> 1, best_pitch);

   /* Refine by pseudo-interpolation */
   if (best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) - 1)
   {
      opus_val32 a, b, c;
      a = xcorr[best_pitch[0] - 1];
      b = xcorr[best_pitch[0]];
      c = xcorr[best_pitch[0] + 1];
      if ((c - a) > MULT16_32_Q15(QCONST16(.7f, 15), b - a))
         offset = 1;
      else if ((a - c) > MULT16_32_Q15(QCONST16(.7f, 15), b - c))
         offset = -1;
      else
         offset = 0;
   }
   else
   {
      offset = 0;
   }
   *pitch = 2 * best_pitch[0] - offset;
}

static opus_val16 compute_pitch_gain(opus_val32 xy, opus_val32 xx, opus_val32 yy)
{
   return xy / sqrt(1 + xx * yy);
}

static const int second_check[16] = {0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2};
opus_val16 rnn_remove_doubling(opus_val16 *x, int maxperiod, int minperiod,
                               int N, int *T0_, int prev_period, opus_val16 prev_gain)
{
   int k, i, T, T0;
   opus_val16 g, g0;
   opus_val16 pg;
   opus_val32 xy, xx, yy, xy2;
   opus_val32 xcorr[3];
   opus_val32 best_xy, best_yy;
   int offset;
   int minperiod0;
   opus_val32 yy_lookup[PITCH_MAX_PERIOD + 1];

   minperiod0 = minperiod;
   maxperiod /= 2;
   minperiod /= 2;
   *T0_ /= 2;
   prev_period /= 2;
   N /= 2;
   x += maxperiod;
   if (*T0_ >= maxperiod)
      *T0_ = maxperiod - 1;

   T = T0 = *T0_;
   dual_inner_prod(x, x, x - T0, N, &xx, &xy);
   yy_lookup[0] = xx;
   yy = xx;
   for (i = 1; i <= maxperiod; i++)
   {
      yy = yy + MULT16_16(x[-i], x[-i]) - MULT16_16(x[N - i], x[N - i]);
      yy_lookup[i] = MAX32(0, yy);
   }
   yy = yy_lookup[T0];
   best_xy = xy;
   best_yy = yy;
   g = g0 = compute_pitch_gain(xy, xx, yy);
   /* Look for any pitch at T/k */
   for (k = 2; k <= 15; k++)
   {
      int T1, T1b;
      opus_val16 g1;
      opus_val16 cont = 0;
      opus_val16 thresh;
      T1 = (2 * T0 + k) / (2 * k);
      if (T1 < minperiod)
         break;
      /* Look for another strong correlation at T1b */
      if (k == 2)
      {
         if (T1 + T0 > maxperiod)
            T1b = T0;
         else
            T1b = T0 + T1;
      }
      else
      {
         T1b = (2 * second_check[k] * T0 + k) / (2 * k);
      }
      dual_inner_prod(x, &x[-T1], &x[-T1b], N, &xy, &xy2);
      xy = HALF32(xy + xy2);
      yy = HALF32(yy_lookup[T1] + yy_lookup[T1b]);
      g1 = compute_pitch_gain(xy, xx, yy);
      if (abs(T1 - prev_period) <= 1)
         cont = prev_gain;
      else if (abs(T1 - prev_period) <= 2 && 5 * k * k < T0)
         cont = HALF16(prev_gain);
      else
         cont = 0;
      thresh = MAX16(QCONST16(.3f, 15), MULT16_16_Q15(QCONST16(.7f, 15), g0) - cont);
      /* Bias against very high pitch (very short period) to avoid false-positives
         due to short-term correlation */
      if (T1 < 3 * minperiod)
         thresh = MAX16(QCONST16(.4f, 15), MULT16_16_Q15(QCONST16(.85f, 15), g0) - cont);
      else if (T1 < 2 * minperiod)
         thresh = MAX16(QCONST16(.5f, 15), MULT16_16_Q15(QCONST16(.9f, 15), g0) - cont);
      if (g1 > thresh)
      {
         best_xy = xy;
         best_yy = yy;
         T = T1;
         g = g1;
      }
   }
   best_xy = MAX32(0, best_xy);
   if (best_yy <= best_xy)
      pg = Q15ONE;
   else
      pg = best_xy / (best_yy + 1);

   for (k = 0; k < 3; k++)
      xcorr[k] = celt_inner_prod(x, x - (T + k - 1), N);
   if ((xcorr[2] - xcorr[0]) > MULT16_32_Q15(QCONST16(.7f, 15), xcorr[1] - xcorr[0]))
      offset = 1;
   else if ((xcorr[0] - xcorr[2]) > MULT16_32_Q15(QCONST16(.7f, 15), xcorr[1] - xcorr[2]))
      offset = -1;
   else
      offset = 0;
   if (pg > g)
      pg = g;
   *T0_ = 2 * T + offset;

   if (*T0_ < minperiod0)
      *T0_ = minperiod0;
   return pg;
}
