#include "kiss_fft.h"

#define MULBYSCALAR(c, s) \
   do                     \
   {                      \
      (c).r *= (s);       \
      (c).i *= (s);       \
   } while (0)

#define MUL(m, a, b)                                       \
   do                                                      \
   {                                                       \
      (m).r = SUB(MULT((a).r, (b).r), MULT((a).i, (b).i)); \
      (m).i = ADD(MULT((a).r, (b).i), MULT((a).i, (b).r)); \
   } while (0)

#define OP(res, a, b, op)         \
   do                             \
   {                              \
      (res).r = op((a).r, (b).r); \
      (res).i = op((a).i, (b).i); \
   } while (0)

#define ADDTO(res, a)                \
   do                                \
   {                                 \
      (res).r = ADD((res).r, (a).r); \
      (res).i = ADD((res).i, (a).i); \
   } while (0)

static void kf_bfly3(kiss_fft_cpx *Fout, const int fstride, const kiss_fft_state *st, int m, int N, int mm)
{
   const int m2 = 2 * m;
   const kiss_fft_cpx *tw1, *tw2;
   kiss_fft_cpx scratch[5];
   kiss_fft_cpx epi3;
   kiss_fft_cpx *Fout_beg = Fout;
   epi3 = st->twiddles[fstride * m];

   for (int i = 0; i < N; i++)
   {
      Fout = Fout_beg + i * mm;
      tw1 = tw2 = st->twiddles;
      /* For non-custom modes, m is guaranteed to be a multiple of 4. */
      int k = m;
      do
      {

         MUL(scratch[1], Fout[m], *tw1);
         MUL(scratch[2], Fout[m2], *tw2);

         OP(scratch[3], scratch[1], scratch[2], ADD);
         OP(scratch[0], scratch[1], scratch[2], SUB);
         tw1 += fstride;
         tw2 += fstride * 2;

         Fout[m].r = SUB(Fout->r, HALF(scratch[3].r));
         Fout[m].i = SUB(Fout->i, HALF(scratch[3].i));

         MULBYSCALAR(scratch[0], epi3.i);

         ADDTO(*Fout, scratch[3]);

         Fout[m2].r = ADD(Fout[m].r, scratch[0].i);
         Fout[m2].i = SUB(Fout[m].i, scratch[0].r);

         Fout[m].r = SUB(Fout[m].r, scratch[0].i);
         Fout[m].i = ADD(Fout[m].i, scratch[0].r);

         ++Fout;
      } while (--k);
   }
}

static void kf_bfly4(kiss_fft_cpx *Fout, const int fstride, const kiss_fft_state *st, int m, int N, int mm)
{
   if (m == 1)
   {
      /* Degenerate case where all the twiddles are 1. */
      for (int i = 0; i < N; i++)
      {
         kiss_fft_cpx scratch0, scratch1;

         OP(scratch0, *Fout, Fout[2], SUB);
         ADDTO(*Fout, Fout[2]);
         OP(scratch1, Fout[1], Fout[3], ADD);
         OP(Fout[2], *Fout, scratch1, SUB);
         ADDTO(*Fout, scratch1);
         OP(scratch1, Fout[1], Fout[3], SUB);

         Fout[1].r = ADD(scratch0.r, scratch1.i);
         Fout[1].i = SUB(scratch0.i, scratch1.r);
         Fout[3].r = SUB(scratch0.r, scratch1.i);
         Fout[3].i = ADD(scratch0.i, scratch1.r);
         Fout += 4;
      }
   }
   else
   {
      kiss_fft_cpx scratch[6];
      const kiss_fft_cpx *tw1, *tw2, *tw3;
      const int m2 = 2 * m;
      const int m3 = 3 * m;
      kiss_fft_cpx *Fout_beg = Fout;
      for (int i = 0; i < N; i++)
      {
         Fout = Fout_beg + i * mm;
         tw3 = tw2 = tw1 = st->twiddles;
         /* m is guaranteed to be a multiple of 4. */
         for (int j = 0; j < m; j++)
         {
            MUL(scratch[0], Fout[m], *tw1);
            MUL(scratch[1], Fout[m2], *tw2);
            MUL(scratch[2], Fout[m3], *tw3);

            OP(scratch[5], *Fout, scratch[1], SUB);
            ADDTO(*Fout, scratch[1]);
            OP(scratch[3], scratch[0], scratch[2], ADD);
            OP(scratch[4], scratch[0], scratch[2], SUB);
            OP(Fout[m2], *Fout, scratch[3], SUB);
            tw1 += fstride;
            tw2 += fstride * 2;
            tw3 += fstride * 3;
            ADDTO(*Fout, scratch[3]);

            Fout[m].r = ADD(scratch[5].r, scratch[4].i);
            Fout[m].i = SUB(scratch[5].i, scratch[4].r);
            Fout[m3].r = SUB(scratch[5].r, scratch[4].i);
            Fout[m3].i = ADD(scratch[5].i, scratch[4].r);
            ++Fout;
         }
      }
   }
}

static void kf_bfly5(kiss_fft_cpx *Fout, const int fstride, const kiss_fft_state *st, int m, int N, int mm)
{
   const kiss_fft_cpx *tw;
   kiss_fft_cpx *Fout0, *Fout1, *Fout2, *Fout3, *Fout4;
   kiss_fft_cpx scratch[13];
   kiss_fft_cpx ya, yb;
   kiss_fft_cpx *Fout_beg = Fout;

   ya = st->twiddles[fstride * m];
   yb = st->twiddles[fstride * 2 * m];
   tw = st->twiddles;

   for (int i = 0; i < N; i++)
   {
      Fout = Fout_beg + i * mm;
      Fout0 = Fout;
      Fout1 = Fout0 + m;
      Fout2 = Fout0 + 2 * m;
      Fout3 = Fout0 + 3 * m;
      Fout4 = Fout0 + 4 * m;

      /* For non-custom modes, m is guaranteed to be a multiple of 4. */
      for (int u = 0; u < m; ++u)
      {
         scratch[0] = *Fout0;

         MUL(scratch[1], *Fout1, tw[u * fstride]);
         MUL(scratch[2], *Fout2, tw[2 * u * fstride]);
         MUL(scratch[3], *Fout3, tw[3 * u * fstride]);
         MUL(scratch[4], *Fout4, tw[4 * u * fstride]);

         OP(scratch[7], scratch[1], scratch[4], ADD);
         OP(scratch[10], scratch[1], scratch[4], SUB);
         OP(scratch[8], scratch[2], scratch[3], ADD);
         OP(scratch[9], scratch[2], scratch[3], SUB);

         Fout0->r = ADD(Fout0->r, ADD(scratch[7].r, scratch[8].r));
         Fout0->i = ADD(Fout0->i, ADD(scratch[7].i, scratch[8].i));

         scratch[5].r = ADD(scratch[0].r, ADD(MULT(scratch[7].r, ya.r), MULT(scratch[8].r, yb.r)));
         scratch[5].i = ADD(scratch[0].i, ADD(MULT(scratch[7].i, ya.r), MULT(scratch[8].i, yb.r)));

         scratch[6].r = ADD(MULT(scratch[10].i, ya.i), MULT(scratch[9].i, yb.i));
         scratch[6].i = NEG(ADD(MULT(scratch[10].r, ya.i), MULT(scratch[9].r, yb.i)));

         OP(*Fout1, scratch[5], scratch[6], SUB);
         OP(*Fout4, scratch[5], scratch[6], ADD);

         scratch[11].r = ADD(scratch[0].r, ADD(MULT(scratch[7].r, yb.r), MULT(scratch[8].r, ya.r)));
         scratch[11].i = ADD(scratch[0].i, ADD(MULT(scratch[7].i, yb.r), MULT(scratch[8].i, ya.r)));
         scratch[12].r = SUB(MULT(scratch[9].i, ya.i), MULT(scratch[10].i, yb.i));
         scratch[12].i = SUB(MULT(scratch[10].r, yb.i), MULT(scratch[9].r, ya.i));

         OP(*Fout2, scratch[11], scratch[12], ADD);
         OP(*Fout3, scratch[11], scratch[12], SUB);

         ++Fout0;
         ++Fout1;
         ++Fout2;
         ++Fout3;
         ++Fout4;
      }
   }
}

void rnn_fft_impl(const kiss_fft_state *st, kiss_fft_cpx *fout)
{
   int m2, m, p, i, L;
   int fstride[MAXFACTORS];
   int shift = st->shift > 0 ? st->shift : 0;

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
      case 3:
         kf_bfly3(fout, fstride[i] << shift, st, m, fstride[i], m2);
         break;
      case 4:
         kf_bfly4(fout, fstride[i] << shift, st, m, fstride[i], m2);
         break;
      case 5:
         kf_bfly5(fout, fstride[i] << shift, st, m, fstride[i], m2);
         break;
      }
      m = m2;
   }
}

void rnn_fft_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
   float_t scale;
   scale = st->scale;
   for (int i = 0; i < st->nfft; i++)
   {
      kiss_fft_cpx x = fin[i];
      fout[st->bitrev[i]].r = MULT(scale, x.r);
      fout[st->bitrev[i]].i = MULT(scale, x.i);
   }
   rnn_fft_impl(st, fout);
}