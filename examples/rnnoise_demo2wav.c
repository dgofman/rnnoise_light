#include <stdio.h>  // fprintf
#include <string.h> // memcmp

#include "../src/denoise.h"

// WAV header structure
typedef struct
{
  char riff[4];
  uint32_t file_size;
  char wave[4];
  char fmt[4];
  uint32_t fmt_size;
  uint16_t audio_format;
  uint16_t num_channels;
  uint32_t sample_rate;
  uint32_t byte_rate;
  uint16_t block_align;
  uint16_t bits_per_sample;
  char data[4];
  uint32_t data_size;
} WavHeader;

// Function to check WAV header validity
static int is_wav_header(const WavHeader *header)
{
  return memcmp(header->riff, "RIFF", 4) == 0 &&
         memcmp(header->wave, "WAVE", 4) == 0 &&
         memcmp(header->fmt, "fmt ", 4) == 0 &&
         memcmp(header->data, "data", 4) == 0 &&
         header->audio_format == 1 &&   // PCM format
         header->bits_per_sample == 16; // 16-bit
}

int main(int argc, char **argv)
{
  int i;
  int first = 1;
  float x[FRAME_SIZE];
  FILE *f1, *fout;
  DenoiseState *st;
  st = rnnoise_create(0);
  if (!st)
  {
    fprintf(stderr, "ERROR: rnnoise_create failed\n");
    return 1;
  }
  printf("DenoiseState initialized successfully\n");

  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <input file> <output wav>\n", argv[0]);
    return 1;
  }

  printf("Input file: %s\n", argv[1]);
  printf("Output file: %s\n", argv[2]);

  f1 = fopen(argv[1], "rb");
  if (!f1)
  {
    fprintf(stderr, "Cannot open %s\n", argv[1]);
    return 1;
  }

  // Check for WAV header
  WavHeader header;
  if (fread(&header, sizeof(header), 1, f1) == 1)
  {
    if (is_wav_header(&header))
    {
      int channels = header.num_channels;
      printf("WAV header detected:\n");
      printf("  Channels: %d\n", channels);
      printf("  Sample rate: %d Hz\n", header.sample_rate);
      printf("  Bits per sample: %d\n", header.bits_per_sample);
      printf("  Data size: %u bytes\n", header.data_size);

      if (channels != 1)
      {
        fprintf(stderr, "WARNING: Please convert to mono using:\n  ffmpeg -i %s -acodec pcm_s16le -ar 48000 -ac 1 -f wav -write_bext 0 -fflags +bitexact %s_mono.wav\n", argv[1], argv[1]);
        fprintf(stderr, "To confirm the result, use:\n  ffprobe %s_mono.wav\n", argv[1]);
      }

      // Handle non-48kHz sample rates
      if (header.sample_rate != 48000)
      {
        printf("WARNING: Input sample rate is %d Hz, but RNNoise requires 48kHz\n", header.sample_rate);
        printf("         Resampling is not implemented - audio may sound incorrect\n");
      }
    }
    else
    {
      // Not a WAV file - rewind to beginning
      fseek(f1, 0, SEEK_SET);
      printf("No valid WAV header found - treating as raw PCM\n");
      printf("Assuming mono, 48kHz, 16-bit PCM\n");
    }
  }
  else
  {
    // Read error - rewind to beginning
    fseek(f1, 0, SEEK_SET);
    printf("No header read - treating as raw PCM\n");
    printf("Assuming mono, 48kHz, 16-bit PCM\n");
  }

  // Create output WAV file
  fout = fopen(argv[2], "wb");
  if (!fout) {
    fprintf(stderr, "Cannot open %s for writing\n", argv[2]);
    fclose(f1);
    return 1;
  }

  // Initialize output WAV header (mono, 16-bit, 48kHz)
  WavHeader out_header = {
    .riff = {'R','I','F','F'},
    .file_size = 0, // placeholder
    .wave = {'W','A','V','E'},
    .fmt = {'f','m','t',' '},
    .fmt_size = 16,
    .audio_format = 1, // PCM
    .num_channels = 1, // mono
    .sample_rate = 48000,
    .byte_rate = 48000 * 2, // sample_rate * channels * bytes_per_sample
    .block_align = 2,       // channels * bytes_per_sample
    .bits_per_sample = 16,
    .data = {'d','a','t','a'},
    .data_size = 0          // placeholder
  };

  // Write initial header (with placeholder sizes)
  fwrite(&out_header, sizeof(out_header), 1, fout);

  long data_start = ftell(fout); // Position after header

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
      tmp[i] = (short)(x[i]);
    if (!first)
      fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
    first = 0;
  }

  // Get output file stats
  long output_size = ftell(fout);
  uint32_t data_size = (uint32_t)(output_size - data_start);

  // Update header with actual sizes
  out_header.file_size = data_size + 36; // RIFF size: 36 + data_size
  out_header.data_size = data_size;

  // Rewind and write updated header
  fseek(fout, 0, SEEK_SET);
  fwrite(&out_header, sizeof(out_header), 1, fout);
  
  rnnoise_destroy(st);
  fclose(f1);
  fclose(fout);
  if (output_size > 0)
  {
    printf("Output file created successfully\n");
    printf("  File size: %ld bytes\n", output_size);
    printf("  Duration: %.2f seconds\n", (float)output_size / (float)(header.num_channels * 2 * 48000));
  }
  else
  {
    printf("ERROR: Output file is empty\n");
    remove(argv[2]); // Delete empty file
  }

  return 0;
}
