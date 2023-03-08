#include <stdint.h>
#include <stdlib.h>
#include "utils/matrix.h"

#ifndef POLICY_PARAMETERS_H_
#define POLICY_PARAMETERS_H_

// #define IS_MSP
#define BITMASK_BYTES 13 // ceil(SEQ_LENGTH/BYTE_SIZE) = ceil(100/8)
#define SEQ_LENGTH 100
#define DEFAULT_WIDTH 16
#define DEFAULT_PRECISION 6
#define TARGET_BYTES 274
#define TARGET_DATA_BYTES 256

#define WINDOWS_PER_DATASET 60

#define MAX_SKIP 4
#define MIN_SKIP 0

#define COLLECTION_RATE 44 // 70% collection rate
// #define BUDGET 865         // 70% of num samples
#define BUDGET 618         // 50% of num samples
#define WINDOW_SIZE 20

#define NUM_INDICES 20
#define OFFSET 1
static size_t IDX_LENGTHS[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
static uint16_t *COLLECT_INDICES[NUM_INDICES] = 
{
    (uint16_t[]) {10},
    (uint16_t[]) {0,10},
    (uint16_t[]) {0,7,13},
    (uint16_t[]) {0,5,10,15},
    (uint16_t[]) {0,4,8,12,16},
    (uint16_t[]) {0,3,7,10,13,16},
    (uint16_t[]) {0,3,6,8,11,14,16},
    (uint16_t[]) {0,2,5,8,11,14,17,19},
    (uint16_t[]) {0,2,4,6,9,11,13,15,18},
    (uint16_t[]) {0,2,4,6,8,10,12,14,16,18},
    (uint16_t[]) {0,2,4,6,8,10,12,14,15,17,19},
    (uint16_t[]) {0,1,3,4,5,7,9,10,12,13,15,17},
    (uint16_t[]) {0,2,3,5,7,8,9,10,12,13,14,15,17},
    (uint16_t[]) {0,1,2,3,5,6,7,9,10,12,14,16,17,19},
    (uint16_t[]) {0,1,2,4,6,7,8,9,10,11,13,15,17,18,19},
    (uint16_t[]) {0,1,2,4,6,7,8,9,10,11,12,14,15,17,18,19},
    (uint16_t[]) {0,1,3,4,5,6,7,9,10,11,12,13,15,16,17,18,19},
    (uint16_t[]) {0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19},
    (uint16_t[]) {0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19},
    (uint16_t[]) {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}
};

// #define IS_ADAPTIVE_HEURISTIC
#ifdef IS_ADAPTIVE_HEURISTIC
#define THRESHOLD 8606
#endif

// #define IS_ADAPTIVE_DEVIATION
// static uint16_t THRESHOLDS[] = {};
#ifdef IS_ADAPTIVE_DEVIATION
#define DEVIATION_PRECISION 6
#define ALPHA 44
#define BETA 44
#define THRESHOLD 6
#endif

#define IS_ADAPTIVE_GAMBLER
#ifdef IS_ADAPTIVE_GAMBLER
#define ALPHA 44
#define BETA 44
#endif

// #define IS_UNIFORM
#ifdef IS_UNIFORM
#define UNIFORM_BUDGET 14 // 70% collection rate
#endif

#endif
