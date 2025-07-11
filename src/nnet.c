#include <string.h> // strcmp
#include <stdlib.h> // calloc, realloc, free
#include "nnet.h"

#define SPARSE_BLOCK_SIZE 32

static int parse_record(const void **data, int *len, WeightArray *array)
{
  WeightHead *h = (WeightHead *)*data;
  if (*len < WEIGHT_BLOCK_SIZE)
    return -1;
  if (h->block_size < h->size)
    return -1;
  if (h->block_size > *len - WEIGHT_BLOCK_SIZE)
    return -1;
  if (h->name[sizeof(h->name) - 1] != 0)
    return -1;
  if (h->size < 0)
    return -1;
  array->name = h->name;
  array->type = h->type;
  array->size = h->size;
  array->data = (void *)((unsigned char *)(*data) + WEIGHT_BLOCK_SIZE);

  *data = (void *)((unsigned char *)*data + h->block_size + WEIGHT_BLOCK_SIZE);
  *len -= h->block_size + WEIGHT_BLOCK_SIZE;
  return array->size;
}

int parse_weights(WeightArray **list, const void *data, int len)
{
  int nb_arrays = 0;
  int capacity = 20;
  *list = calloc(capacity * sizeof(WeightArray), 1);
  while (len > 0)
  {
    int ret;
    WeightArray array = {NULL, 0, 0, 0};
    ret = parse_record(&data, &len, &array);
    if (ret > 0)
    {
      if (nb_arrays + 1 >= capacity)
      {
        /* Make sure there's room for the ending NULL element too. */
        capacity = capacity * 3 / 2;
        *list = realloc(*list, capacity * sizeof(WeightArray));
      }
      (*list)[nb_arrays++] = array;
    }
    else
    {
      free(*list);
      *list = NULL;
      return -1;
    }
  }
  (*list)[nb_arrays].name = NULL;
  return nb_arrays;
}

static const void *find_array_entry(const WeightArray *arrays, const char *name)
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

static const void *opt_array_check(const WeightArray *arrays, const char *name, int size, int *error)
{
  const WeightArray *a = find_array_entry(arrays, name);
  *error = (a->name != NULL && a->size != size);
  if (a->name && a->size == size)
    return a->data;
  else
    return NULL;
}

static const void *find_idx_check(const WeightArray *arrays, const char *name, int nb_in, int nb_out, int *total_blocks)
{
  int remain;
  const int *idx;
  const WeightArray *a = find_array_entry(arrays, name);
  *total_blocks = 0;
  if (a == NULL)
    return NULL;
  idx = a->data;
  remain = a->size / sizeof(int);
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
  return a->data;
}

int linear_init(LinearLayer *layer, const WeightArray *arrays,
                const char *bias,
                const char *subias,
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
  layer->subias = NULL;
  layer->weights = NULL;
  layer->float_weights = NULL;
  layer->weights_idx = NULL;
  layer->diag = NULL;
  layer->scale = NULL;
  if (bias != NULL)
  {
    if ((layer->bias = find_array_check(arrays, bias, nb_outputs * sizeof(layer->bias[0]))) == NULL)
      return 1;
  }
  if (subias != NULL)
  {
    if ((layer->subias = find_array_check(arrays, subias, nb_outputs * sizeof(layer->subias[0]))) == NULL)
      return 1;
  }
  if (weights_idx != NULL)
  {
    int total_blocks;
    if ((layer->weights_idx = find_idx_check(arrays, weights_idx, nb_inputs, nb_outputs, &total_blocks)) == NULL)
      return 1;
    if (weights != NULL)
    {
      if ((layer->weights = find_array_check(arrays, weights, SPARSE_BLOCK_SIZE * total_blocks * sizeof(layer->weights[0]))) == NULL)
        return 1;
    }
    if (float_weights != NULL)
    {
      layer->float_weights = opt_array_check(arrays, float_weights, SPARSE_BLOCK_SIZE * total_blocks * sizeof(layer->float_weights[0]), &err);
      if (err)
        return 1;
    }
  }
  else
  {
    if (weights != NULL)
    {
      if ((layer->weights = find_array_check(arrays, weights, nb_inputs * nb_outputs * sizeof(layer->weights[0]))) == NULL)
        return 1;
    }
    if (float_weights != NULL)
    {
      layer->float_weights = opt_array_check(arrays, float_weights, nb_inputs * nb_outputs * sizeof(layer->float_weights[0]), &err);
      if (err)
        return 1;
    }
  }
  if (diag != NULL)
  {
    if ((layer->diag = find_array_check(arrays, diag, nb_outputs * sizeof(layer->diag[0]))) == NULL)
      return 1;
  }
  if (weights != NULL)
  {
    if ((layer->scale = find_array_check(arrays, scale, nb_outputs * sizeof(layer->scale[0]))) == NULL)
      return 1;
  }
  layer->nb_inputs = nb_inputs;
  layer->nb_outputs = nb_outputs;
  return 0;
}