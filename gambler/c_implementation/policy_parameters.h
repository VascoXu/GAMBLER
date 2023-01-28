#include <stdint.h>
#include "utils/matrix.h"

#ifndef POLICY_PARAMETERS_H_
#define POLICY_PARAMETERS_H_
// #define IS_MSP
#define BITMASK_BYTES 7
#define SEQ_LENGTH 50
#define NUM_FEATURES 6
#define DEFAULT_WIDTH 16
#define DEFAULT_PRECISION 13
#define TARGET_BYTES 274
#define TARGET_DATA_BYTES 256

#define MAX_SKIP 4
#define MIN_SKIP 0

// #define IS_ADAPTIVE_HEURISTIC
#ifdef IS_ADAPTIVE_HEURISTIC
#define THRESHOLD 8606
#endif

// #define IS_UNIFORM
#ifdef IS_UNIFORM
#define NUM_INDICES 20
static const uint16_t COLLECT_INDICES[NUM_INDICES] = {0,3,6,9,11,13,16,18,21,23,25,28,30,32,34,37,40,43,45,48};
#endif

#define IS_ADAPTIVE_DEVIATION
#ifdef IS_ADAPTIVE_DEVIATION
#define DEVIATION_PRECISION 5
#define ALPHA 22
#define BETA 22
#define THRESHOLD 350
#endif

#endif
