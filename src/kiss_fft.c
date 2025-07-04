#include "kiss_fft.h"

void kf_bfly2(
    kiss_fft_cpx *Fout,
    int m,
    int N)
{
   kiss_fft_cpx *Fout2;
   int i;
   (void)m;
   {
      opus_val16 tw;
      tw = QCONST16(0.7071067812f, 15);
      for (i = 0; i < N; i++)
      {
         kiss_fft_cpx t;
         Fout2 = Fout + 4;
         t = Fout2[0];
         C_SUB(Fout2[0], Fout[0], t);
         C_ADDTO(Fout[0], t);

         t.r = S_MUL(ADD32_ovflw(Fout2[1].r, Fout2[1].i), tw);
         t.i = S_MUL(SUB32_ovflw(Fout2[1].i, Fout2[1].r), tw);
         C_SUB(Fout2[1], Fout[1], t);
         C_ADDTO(Fout[1], t);

         t.r = Fout2[2].i;
         t.i = -Fout2[2].r;
         C_SUB(Fout2[2], Fout[2], t);
         C_ADDTO(Fout[2], t);

         t.r = S_MUL(SUB32_ovflw(Fout2[3].i, Fout2[3].r), tw);
         t.i = S_MUL(NEG32_ovflw(ADD32_ovflw(Fout2[3].i, Fout2[3].r)), tw);
         C_SUB(Fout2[3], Fout[3], t);
         C_ADDTO(Fout[3], t);
         Fout += 8;
      }
   }
}

void kf_bfly3(
    kiss_fft_cpx *Fout,
    const size_t fstride,
    const kiss_fft_state *st,
    int m,
    int N,
    int mm)
{
   int i;
   size_t k;
   const size_t m2 = 2 * m;
   const kiss_twiddle_cpx *tw1, *tw2;
   kiss_fft_cpx scratch[5];
   kiss_twiddle_cpx epi3;

   kiss_fft_cpx *Fout_beg = Fout;
   epi3 = st->twiddles[fstride * m];

   for (i = 0; i < N; i++)
   {
      Fout = Fout_beg + i * mm;
      tw1 = tw2 = st->twiddles;
      /* For non-custom modes, m is guaranteed to be a multiple of 4. */
      k = m;
      do
      {

         C_MUL(scratch[1], Fout[m], *tw1);
         C_MUL(scratch[2], Fout[m2], *tw2);

         C_ADD(scratch[3], scratch[1], scratch[2]);
         C_SUB(scratch[0], scratch[1], scratch[2]);
         tw1 += fstride;
         tw2 += fstride * 2;

         Fout[m].r = SUB32_ovflw(Fout->r, HALF_OF(scratch[3].r));
         Fout[m].i = SUB32_ovflw(Fout->i, HALF_OF(scratch[3].i));

         C_MULBYSCALAR(scratch[0], epi3.i);

         C_ADDTO(*Fout, scratch[3]);

         Fout[m2].r = ADD32_ovflw(Fout[m].r, scratch[0].i);
         Fout[m2].i = SUB32_ovflw(Fout[m].i, scratch[0].r);

         Fout[m].r = SUB32_ovflw(Fout[m].r, scratch[0].i);
         Fout[m].i = ADD32_ovflw(Fout[m].i, scratch[0].r);

         ++Fout;
      } while (--k);
   }
}

void kf_bfly4(
    kiss_fft_cpx *Fout,
    const size_t fstride,
    const kiss_fft_state *st,
    int m,
    int N,
    int mm)
{
   int i;

   if (m == 1)
   {
      /* Degenerate case where all the twiddles are 1. */
      for (i = 0; i < N; i++)
      {
         kiss_fft_cpx scratch0, scratch1;

         C_SUB(scratch0, *Fout, Fout[2]);
         C_ADDTO(*Fout, Fout[2]);
         C_ADD(scratch1, Fout[1], Fout[3]);
         C_SUB(Fout[2], *Fout, scratch1);
         C_ADDTO(*Fout, scratch1);
         C_SUB(scratch1, Fout[1], Fout[3]);

         Fout[1].r = ADD32_ovflw(scratch0.r, scratch1.i);
         Fout[1].i = SUB32_ovflw(scratch0.i, scratch1.r);
         Fout[3].r = SUB32_ovflw(scratch0.r, scratch1.i);
         Fout[3].i = ADD32_ovflw(scratch0.i, scratch1.r);
         Fout += 4;
      }
   }
   else
   {
      int j;
      kiss_fft_cpx scratch[6];
      const kiss_twiddle_cpx *tw1, *tw2, *tw3;
      const int m2 = 2 * m;
      const int m3 = 3 * m;
      kiss_fft_cpx *Fout_beg = Fout;
      for (i = 0; i < N; i++)
      {
         Fout = Fout_beg + i * mm;
         tw3 = tw2 = tw1 = st->twiddles;
         /* m is guaranteed to be a multiple of 4. */
         for (j = 0; j < m; j++)
         {
            C_MUL(scratch[0], Fout[m], *tw1);
            C_MUL(scratch[1], Fout[m2], *tw2);
            C_MUL(scratch[2], Fout[m3], *tw3);

            C_SUB(scratch[5], *Fout, scratch[1]);
            C_ADDTO(*Fout, scratch[1]);
            C_ADD(scratch[3], scratch[0], scratch[2]);
            C_SUB(scratch[4], scratch[0], scratch[2]);
            C_SUB(Fout[m2], *Fout, scratch[3]);
            tw1 += fstride;
            tw2 += fstride * 2;
            tw3 += fstride * 3;
            C_ADDTO(*Fout, scratch[3]);

            Fout[m].r = ADD32_ovflw(scratch[5].r, scratch[4].i);
            Fout[m].i = SUB32_ovflw(scratch[5].i, scratch[4].r);
            Fout[m3].r = SUB32_ovflw(scratch[5].r, scratch[4].i);
            Fout[m3].i = ADD32_ovflw(scratch[5].i, scratch[4].r);
            ++Fout;
         }
      }
   }
}

void kf_bfly5(
    kiss_fft_cpx *Fout,
    const size_t fstride,
    const kiss_fft_state *st,
    int m,
    int N,
    int mm)
{
   kiss_fft_cpx *Fout0, *Fout1, *Fout2, *Fout3, *Fout4;
   int i, u;
   kiss_fft_cpx scratch[13];
   const kiss_twiddle_cpx *tw;
   kiss_twiddle_cpx ya, yb;
   kiss_fft_cpx *Fout_beg = Fout;

   ya = st->twiddles[fstride * m];
   yb = st->twiddles[fstride * 2 * m];

   tw = st->twiddles;

   for (i = 0; i < N; i++)
   {
      Fout = Fout_beg + i * mm;
      Fout0 = Fout;
      Fout1 = Fout0 + m;
      Fout2 = Fout0 + 2 * m;
      Fout3 = Fout0 + 3 * m;
      Fout4 = Fout0 + 4 * m;

      /* For non-custom modes, m is guaranteed to be a multiple of 4. */
      for (u = 0; u < m; ++u)
      {
         scratch[0] = *Fout0;

         C_MUL(scratch[1], *Fout1, tw[u * fstride]);
         C_MUL(scratch[2], *Fout2, tw[2 * u * fstride]);
         C_MUL(scratch[3], *Fout3, tw[3 * u * fstride]);
         C_MUL(scratch[4], *Fout4, tw[4 * u * fstride]);

         C_ADD(scratch[7], scratch[1], scratch[4]);
         C_SUB(scratch[10], scratch[1], scratch[4]);
         C_ADD(scratch[8], scratch[2], scratch[3]);
         C_SUB(scratch[9], scratch[2], scratch[3]);

         Fout0->r = ADD32_ovflw(Fout0->r, ADD32_ovflw(scratch[7].r, scratch[8].r));
         Fout0->i = ADD32_ovflw(Fout0->i, ADD32_ovflw(scratch[7].i, scratch[8].i));

         scratch[5].r = ADD32_ovflw(scratch[0].r, ADD32_ovflw(S_MUL(scratch[7].r, ya.r), S_MUL(scratch[8].r, yb.r)));
         scratch[5].i = ADD32_ovflw(scratch[0].i, ADD32_ovflw(S_MUL(scratch[7].i, ya.r), S_MUL(scratch[8].i, yb.r)));

         scratch[6].r = ADD32_ovflw(S_MUL(scratch[10].i, ya.i), S_MUL(scratch[9].i, yb.i));
         scratch[6].i = NEG32_ovflw(ADD32_ovflw(S_MUL(scratch[10].r, ya.i), S_MUL(scratch[9].r, yb.i)));

         C_SUB(*Fout1, scratch[5], scratch[6]);
         C_ADD(*Fout4, scratch[5], scratch[6]);

         scratch[11].r = ADD32_ovflw(scratch[0].r, ADD32_ovflw(S_MUL(scratch[7].r, yb.r), S_MUL(scratch[8].r, ya.r)));
         scratch[11].i = ADD32_ovflw(scratch[0].i, ADD32_ovflw(S_MUL(scratch[7].i, yb.r), S_MUL(scratch[8].i, ya.r)));
         scratch[12].r = SUB32_ovflw(S_MUL(scratch[9].i, ya.i), S_MUL(scratch[10].i, yb.i));
         scratch[12].i = SUB32_ovflw(S_MUL(scratch[10].r, yb.i), S_MUL(scratch[9].r, ya.i));

         C_ADD(*Fout2, scratch[11], scratch[12]);
         C_SUB(*Fout3, scratch[11], scratch[12]);

         ++Fout0;
         ++Fout1;
         ++Fout2;
         ++Fout3;
         ++Fout4;
      }
   }
}
