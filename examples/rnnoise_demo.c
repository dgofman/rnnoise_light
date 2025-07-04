#include <stdio.h>
#include "../src/denoise.h"

#define FRAME_SIZE 480

int main(int argc, char **argv)
{
  int i;
  int first = 1;
  float x[FRAME_SIZE];
  FILE *f1, *fout;
  DenoiseState *st;
  st = rnnoise_create(NULL);
  if (!st)
  {
    fprintf(stderr, "rnnoise_create failed\n");
    return 1;
  }

  if (argc != 3)
  {
    fprintf(stderr, "usage: %s <noisy speech> <output denoised>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "rb");
  fout = fopen(argv[2], "wb");
  if (!f1)
  {
    fprintf(stderr, "Cannot open %s\n", argv[1]);
    return 1;
  }
  while (1)
  {
    short tmp[FRAME_SIZE];
    fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1))
      break;
    for (i = 0; i < FRAME_SIZE; i++)
      x[i] = tmp[i];
    rnnoise_process_frame(st, x, x);
    for (i = 0; i < FRAME_SIZE; i++)
      tmp[i] = x[i];
    if (!first)
      fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
    first = 0;
  }
  rnnoise_destroy(st);
  fclose(f1);
  fclose(fout);

  return 0;
}
