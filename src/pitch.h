#ifndef PITCH_H
#define PITCH_H

#include "arch.h"

#define PITCH_MIN_PERIOD 60

#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD + PITCH_FRAME_SIZE)

void rnn_pitch_downsample(celt_sig *x[], opus_val16 *x_lp,
                          int len, int C);

void rnn_pitch_search(const opus_val16 *x_lp, opus_val16 *y,
                      int len, int max_pitch, int *pitch);

opus_val16 rnn_remove_doubling(opus_val16 *x, int maxperiod, int minperiod,
                               int N, int *T0, int prev_period, opus_val16 prev_gain);

static inline void xcorr_kernel(const opus_val16 *x, const opus_val16 *y, opus_val32 sum[4], int len)
{
   int j;
   opus_val16 y_0, y_1, y_2, y_3;
   y_3 = 0; /* gcc doesn't realize that y_3 can't be used uninitialized */
   y_0 = *y++;
   y_1 = *y++;
   y_2 = *y++;
   for (j = 0; j < len - 3; j += 4)
   {
      opus_val16 tmp;
      tmp = *x++;
      y_3 = *y++;
      sum[0] = MAC16_16(sum[0], tmp, y_0);
      sum[1] = MAC16_16(sum[1], tmp, y_1);
      sum[2] = MAC16_16(sum[2], tmp, y_2);
      sum[3] = MAC16_16(sum[3], tmp, y_3);
      tmp = *x++;
      y_0 = *y++;
      sum[0] = MAC16_16(sum[0], tmp, y_1);
      sum[1] = MAC16_16(sum[1], tmp, y_2);
      sum[2] = MAC16_16(sum[2], tmp, y_3);
      sum[3] = MAC16_16(sum[3], tmp, y_0);
      tmp = *x++;
      y_1 = *y++;
      sum[0] = MAC16_16(sum[0], tmp, y_2);
      sum[1] = MAC16_16(sum[1], tmp, y_3);
      sum[2] = MAC16_16(sum[2], tmp, y_0);
      sum[3] = MAC16_16(sum[3], tmp, y_1);
      tmp = *x++;
      y_2 = *y++;
      sum[0] = MAC16_16(sum[0], tmp, y_3);
      sum[1] = MAC16_16(sum[1], tmp, y_0);
      sum[2] = MAC16_16(sum[2], tmp, y_1);
      sum[3] = MAC16_16(sum[3], tmp, y_2);
   }
   if (j++ < len)
   {
      opus_val16 tmp = *x++;
      y_3 = *y++;
      sum[0] = MAC16_16(sum[0], tmp, y_0);
      sum[1] = MAC16_16(sum[1], tmp, y_1);
      sum[2] = MAC16_16(sum[2], tmp, y_2);
      sum[3] = MAC16_16(sum[3], tmp, y_3);
   }
   if (j++ < len)
   {
      opus_val16 tmp = *x++;
      y_0 = *y++;
      sum[0] = MAC16_16(sum[0], tmp, y_1);
      sum[1] = MAC16_16(sum[1], tmp, y_2);
      sum[2] = MAC16_16(sum[2], tmp, y_3);
      sum[3] = MAC16_16(sum[3], tmp, y_0);
   }
   if (j < len)
   {
      opus_val16 tmp = *x++;
      y_1 = *y++;
      sum[0] = MAC16_16(sum[0], tmp, y_2);
      sum[1] = MAC16_16(sum[1], tmp, y_3);
      sum[2] = MAC16_16(sum[2], tmp, y_0);
      sum[3] = MAC16_16(sum[3], tmp, y_1);
   }
}

static inline void dual_inner_prod(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
                                   int N, opus_val32 *xy1, opus_val32 *xy2)
{
   int i;
   opus_val32 xy01 = 0;
   opus_val32 xy02 = 0;
   for (i = 0; i < N; i++)
   {
      xy01 = MAC16_16(xy01, x[i], y01[i]);
      xy02 = MAC16_16(xy02, x[i], y02[i]);
   }
   *xy1 = xy01;
   *xy2 = xy02;
}

static inline opus_val32 celt_inner_prod(const opus_val16 *x,
                                         const opus_val16 *y, int N)
{
   int i;
   opus_val32 xy = 0;
   for (i = 0; i < N; i++)
      xy = MAC16_16(xy, x[i], y[i]);
   return xy;
}

void rnn_pitch_xcorr(const opus_val16 *_x, const opus_val16 *_y,
                     opus_val32 *xcorr, int len, int max_pitch);

#endif
