#include <stdint.h>

/**
 * This is a precomputed half Hann window used to taper the input signal
 * before applying the FFT (Fast Fourier Transform). 
 * It helps reduce spectral leakage by smoothly transitioning the signal edges to zero.
 */
const float rnn_half_window[10] = {
    0.000000f, 0.116978f, 0.413176f, 0.750000f, 0.969846f, 0.969846f, 0.750000f, 0.413176f, 0.116978f, 0.000000f
};

/**
 * This is a precomputed bit-reversal lookup table used in radix-2 FFT algorithms
 * to efficiently reorder the input samples before performing the FFT.
 * Each entry maps an index to its reversed binary form.
 */
const int32_t rnn_kfft[10] = {
    0, 4, 2, 6, 1, 5, 3, 7, 1, 9
};

/**
 * This is a precomputed Discrete Cosine Transform (DCT) matrix used to
 * transform time-domain features into the frequency domain. This is commonly
 * used for feature extraction in neural audio processing.
 */
const float rnn_dct_table[100] = {
    1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f,
};
