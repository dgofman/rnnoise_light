#ifndef ARCH_H
#define ARCH_H

#define Q15ONE 1.0f

#define ROUND16(a, shift) (a)
#define HALF16(x) (.5f * (x))
#define EXTRACT16(x) (x)
#define QCONST16(x, bits) (x)
#define MIN16(a, b) ((a) < (b) ? (a) : (b))
#define MAX16(a, b) ((a) > (b) ? (a) : (b))

#define MAC16_16(c, a, b) ((c) + (opus_val32)(a) * (opus_val32)(b))
#define MULT16_16(a, b) ((opus_val32)(a) * (opus_val32)(b))
#define MULT16_16_Q15(a, b) ((a) * (b))

#define MULT16_32_Q15(a, b) ((a) * (b))
#define MULT16_32_Q16(a, b) ((a) * (b))

#define MULT32_32_Q31(a, b) ((a) * (b))

#define ADD32(a, b) ((a) + (b))
#define HALF32(x) (.5f * (x))
#define SHR32(a, shift) (a)
#define SHL32(a, shift) (a)
#define VSHR32(a, shift) (a)
#define EXTEND32(x) (x)
#define MIN32(a, b) ((a) < (b) ? (a) : (b))
#define MAX32(a, b) ((a) > (b) ? (a) : (b))

#define ADD32_ovflw(a, b) ((a) + (b))
#define SUB32_ovflw(a, b) ((a) - (b))
#define NEG32_ovflw(x) (-(x))

typedef float opus_val16;
typedef float opus_val32;
typedef float celt_sig;

#endif /* ARCH_H */
