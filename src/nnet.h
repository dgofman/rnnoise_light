#ifndef NNET_H_
#define NNET_H_

#define WEIGHT_BLOCK_SIZE 64

typedef struct
{
  const char *name;
  int type;
  int size;
  const void *data;
} WeightArray;

typedef struct
{
  char head[4];
  int version;
  int type;
  int size;
  int block_size;
  char name[44];
} WeightHead;

typedef struct
{
  const float *bias;
  const float *subias;
  const signed char *weights;
  const float *float_weights;
  const int *weights_idx;
  const float *diag;
  const float *scale;
  int nb_inputs;
  int nb_outputs;
} LinearLayer;

int parse_weights(WeightArray **list, const void *data, int len);

int linear_init(LinearLayer *layer, const WeightArray *arrays,
                const char *bias,
                const char *subias,
                const char *weights,
                const char *float_weights,
                const char *weights_idx,
                const char *diag,
                const char *scale,
                int nb_inputs,
                int nb_outputs);

#endif /* NNET_H_ */