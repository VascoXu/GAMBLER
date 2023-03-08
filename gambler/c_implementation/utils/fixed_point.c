#include "fixed_point.h"


FixedPoint fp_add(FixedPoint x, FixedPoint y) {
    return x + y;
}


FixedPoint fp_sub(FixedPoint x, FixedPoint y) {
    return x - y;
}


FixedPoint fp_mul(FixedPoint x, FixedPoint y, uint16_t precision) {
    return (x * y) >> precision;
}

FixedPoint fp_div(FixedPoint x, FixedPoint y, uint16_t precision) {
    int32_t xLarge = ((int32_t) x) << precision;
    return (int16_t) (xLarge / y);
}


FixedPoint fp_neg(FixedPoint x) {
    return -1 * x;
}


FixedPoint fp_abs(FixedPoint x) {
    return x * ((x > 0) - (x < 0));
}


FixedPoint fp_convert(FixedPoint x, uint16_t oldPrecision, uint16_t newPrecision, uint16_t newWidth) {
    int16_t diff = oldPrecision - newPrecision;

    FixedPoint result;
    if (diff >= 0) {
        result = x >> diff;
    } else {
        diff *= -1;
        result = x << diff;
    }

    uint16_t mask = (1 << newWidth) - 1;
    return result & mask;
}

int16_t float_to_fp(float x, uint16_t precision) {
    return (int16_t) (x * (1 << precision));
}


int16_t int_to_fp(int16_t x, uint16_t precision) {
    return x * (1 << precision);
}




void fp_convert_array(FixedPoint *array, uint16_t oldPrecision, uint16_t newPrecision, uint16_t newWidth, uint16_t startIdx, uint16_t length) {
    uint16_t i;
    for (i = startIdx; i < startIdx + length; i++) {
        array[i] = fp_convert(array[i], oldPrecision, newPrecision, newWidth);
    }
}


FixedPoint fp_gated_add_scalar(FixedPoint fp1, FixedPoint fp2, FixedPoint gate, uint16_t precision) {
    /**
     * Returns a vector with gate * fp1 + (1 - gate) * fp2
     */
    uint16_t i, j;
    FixedPoint temp1, temp2, result;

    FixedPoint oneMinusGate = fp_sub(1 << precision, gate);

    temp1 = fp32_mul(fp1, gate, precision);
    temp2 = fp32_mul(fp2, oneMinusGate, precision);
    result = fp_add(temp1, temp2);

    return result;
}



//FixedPoint fp_sigmoid(FixedPoint x, uint16_t precision) {
//    /**
//     * Approximates a sigmoid function using linear components.
//     */
//    FixedPoint one = 1 << precision;
//    FixedPoint two = 1 << (precision + 1);
//    FixedPoint three = one + two;
//
//    FixedPoint result = one;
//    if (x < (-1 * three)) {
//        result = 0;
//    } else if (x < (-1 * one)) {
//        result = (x >> 3) + (5 * (1 << (precision - 4)));
//    } else if (x < one) {
//        result = (x >> 2) + (1 << (precision - 1));
//    } else if (x < three) {
//        result = (x >> 3) + (5 * (1 << (precision - 3)));
//    }
//
//    return result;
//}
//
//
//FixedPoint fp_tanh(FixedPoint x, uint16_t precision) {
//    /**
//     * Approximates a tanh function using a sigmoid.
//     */
//    FixedPoint sigmoidX = fp_sigmoid(x << 1, precision);
//    return (sigmoidX << 1) - (1 << precision);
//}

int16_t fp_tanh(int16_t x, uint16_t precision) {
    /**
     * Approximates tanh using a polynomial.
     */
    uint8_t shouldInvertSign = 0;
    if (x < 0) {
        x = fp_neg(x);
        shouldInvertSign = 1;
    }

    FixedPoint fourth = 1 << (precision - 2);
    FixedPoint half = 1 << (precision - 1);
    FixedPoint one = 1 << precision;
    FixedPoint two = 1 << (precision + 1);

    // Approximate tanh(x) using a piece-wise linear function
    FixedPoint result = one;
    if (x <= fourth) {
        result = x;
    } else if (x <= 3 * fourth) {
        result = 3 * (x >> 2) + 5 * (1 << (precision - 6));
    } else if (x <= (one + fourth)) {
        result = (x >> 1) + fourth;
    } else if (x <= (two + fourth)) {
        result = (x >> 3) + (half + fourth - (1 << (precision - 5)));

        if (result > one) {
            result = one;
        }
    }

    if (shouldInvertSign) {
        return fp_neg(result);
    }
    return result;
}


int16_t fp_sigmoid(int16_t x, uint16_t precision) {
    /**
     * Approximates the sigmoid function using tanh.
     */
    uint8_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }

    FixedPoint one = 1 << precision;
    FixedPoint tanh = fp_tanh(x >> 1, precision);
    FixedPoint result = fp_add(tanh, one) >> 1;

    if (should_invert_sign) {
        result = one - result;
    }

    return result;
}


// 32 bit fixed-point operations
int32_t fp32_add(int32_t x, int32_t y) {
    return x + y;
}


int32_t fp32_neg(int32_t x) {
    return -1 * x;
}


int32_t fp32_sub(int32_t x, int32_t y) {
    return fp32_add(x, fp32_neg(y));
}


int32_t fp32_mul(int32_t x, int32_t y, uint16_t precision) {
    int64_t xLarge = (int64_t) x;
    int64_t yLarge = (int64_t) y;

    return (int32_t) ((xLarge * yLarge) >> precision);
}


int32_t fp32_div(int32_t x, int32_t y, uint16_t precision) {
    int64_t xLarge = ((int64_t) x) << precision;
    return (int32_t) (xLarge / y);
}
