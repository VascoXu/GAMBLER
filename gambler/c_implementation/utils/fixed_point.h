#include <stdint.h>
#include <stdio.h>

#ifndef FIXED_POINT_H_
#define FIXED_POINT_H_

#define INT_TO_FIXED(x) ((x) << DEFAULT_PRECISION)
#define FIXED_TO_INT(x) ((x) >> DEFAULT_PRECISION)
#define FIXED_TO_DOUBLE(x) (((double) (x)) / (1 << DEFAULT_PRECISION))

typedef int16_t FixedPoint;

FixedPoint fp_add(FixedPoint x, FixedPoint y);
FixedPoint fp_mul(FixedPoint x, FixedPoint y, uint16_t precision);
FixedPoint fp_div(FixedPoint x, FixedPoint y, uint16_t precision);
FixedPoint fp_abs(FixedPoint x);
FixedPoint fp_sub(FixedPoint x, FixedPoint y);
FixedPoint fp_neg(FixedPoint x);
FixedPoint fp_sigmoid(FixedPoint x, uint16_t precision);
FixedPoint fp_tanh(FixedPoint x, uint16_t precision);
FixedPoint fp_convert(FixedPoint x, uint16_t oldPrecision, uint16_t newWidth, uint16_t newPrecision);
FixedPoint fp_gated_add_scalar(FixedPoint fp1, FixedPoint fp2, FixedPoint gate, uint16_t precision);

int16_t float_to_fp(float x, uint16_t precision);
int16_t int_to_fp(int16_t x, uint16_t precision);

// 32 bit fixed point operations for improved precision. These are slightly more expensive
// on 16 bit MCUs
int32_t fp32_add(int32_t x, int32_t y);
int32_t fp32_neg(int32_t x);
int32_t fp32_sub(int32_t x, int32_t y);
int32_t fp32_mul(int32_t x, int32_t y, uint16_t precision);
int32_t fp32_div(int32_t x, int32_t y, uint16_t precision);
int32_t fp32_sqrt(int32_t x, uint16_t precision);
int32_t int_to_fp32(int32_t x, uint16_t precision);

void fp_convert_array(FixedPoint *array, uint16_t oldPrecision, uint16_t newPrecision, uint16_t newWidth, uint16_t startIdx, uint16_t length);

#endif

