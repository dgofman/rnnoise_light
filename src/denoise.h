#ifndef RNNOISE_H
#define RNNOISE_H 1

#include <stdio.h>  // FILE
#include <stdint.h> // int16_t, int32_t
#include <string.h> // memcpy
#include "kiss_fft.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define FRAME_SIZE 480
#define FREQ_SIZE (FRAME_SIZE + 1)
#define WINDOW_SIZE (2 * FRAME_SIZE)

#define NB_BANDS 32
#define NB_FEATURES (2 * NB_BANDS + 1)

#define MAXFACTORS 8

   typedef struct DenoiseState DenoiseState;
   typedef struct RNNModel RNNModel;

   /**
    * Allocate and initialize a DenoiseState
    *
    * If model is NULL the default model is used.
    *
    * The returned pointer MUST be freed with rnnoise_destroy().
    */
   DenoiseState *rnnoise_create(RNNModel *model);

   /**
    * Denoise a frame of samples
    *
    */
   float rnnoise_process_frame(DenoiseState *st, float *out, const float *in);

   /**
    * Free a DenoiseState produced by rnnoise_create.
    *
    * The optional custom model must be freed by rnnoise_model_free() after.
    */
   void rnnoise_destroy(DenoiseState *st);

   /**
    * Load a model from a file
    *
    * It must be deallocated with rnnoise_model_free() and the file must not be
    * closed until the returned object is destroyed.
    */
   RNNModel *rnnoise_model_from_file(FILE *f);

   /**
    * Free a custom model
    *
    * It must be called after all the DenoiseStates referring to it are freed.
    */
   void rnnoise_model_free(RNNModel *model);

   /**
    * Return the size of DenoiseState
    */
   int rnnoise_get_size(void);

   /**
    * Initializes a pre-allocated DenoiseState
    *
    * If model is NULL the default model is used.
    *
    * See: rnnoise_create() and rnnoise_model_from_file()
    */
   int rnnoise_init(DenoiseState *st, RNNModel *model);

#ifdef __cplusplus
}
#endif

#endif
