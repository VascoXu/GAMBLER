#include <stdint.h>
#include <stdio.h>

#include "utils/fixed_point.h"
#include "utils/matrix.h"
#include "parameters.h"
#include "data.h"
#include "policy_parameters.h"

#ifndef POLICY_H_
#define POLICY_H_

struct UniformPolicy {
    uint16_t collectIdx;
    uint16_t **collectIndices;
    uint16_t numIndices;
};

struct HeuristicPolicy {
    uint16_t maxSkip;
    uint16_t minSkip;
    uint16_t currentSkip;
    uint16_t sampleSkip;
    FixedPoint threshold;
};

struct DeviationPolicy {
    uint16_t maxSkip;
    uint16_t minSkip;
    uint16_t currentSkip;
    uint16_t sampleSkip;
    FixedPoint threshold;
    FixedPoint alpha;
    FixedPoint beta;
    struct Vector *mean;
    struct Vector *dev;
    uint16_t precision;
};

struct GamblerPolicy {
    uint16_t collectIdx;
    FixedPoint collectionRate;
    uint16_t budget;
    uint16_t samplesLeft;
    uint16_t collected;
    uint16_t **collectIndices;
    FixedPoint *predFeatures;
    uint16_t numIndices;
    FixedPoint alpha;
    FixedPoint beta;
    struct Vector *mean;
    struct Vector *dev;
    FixedPoint meanDev;
    uint16_t windowSize;
    uint16_t precision;
};

// Uniform Policy Operations
void uniform_policy_init(struct UniformPolicy *policy, uint16_t *collectIndices[NUM_INDICES], uint16_t numIndices);
uint8_t uniform_should_collect(struct UniformPolicy *policy, uint16_t seqIdx);
void uniform_update(struct UniformPolicy *policy);
void uniform_reset(struct UniformPolicy *policy);

// Heuristic Policy Operations
void heuristic_policy_init(struct HeuristicPolicy *policy, uint16_t maxSkip, uint16_t minSkip, FixedPoint threshold);
uint8_t heuristic_should_collect(struct HeuristicPolicy *policy, uint16_t seqIdx);
void heuristic_update(struct HeuristicPolicy *policy, struct Vector *curr, struct Vector *prev);
void heuristic_reset(struct HeuristicPolicy *policy);

// Deviation Policy Operations
void deviation_policy_init(struct DeviationPolicy *policy, uint16_t maxSkip, uint16_t minSkip, FixedPoint threshold, FixedPoint alpha, FixedPoint beta, struct Vector *mean, struct Vector *dev);
uint8_t deviation_should_collect(struct DeviationPolicy *policy, uint16_t seqIdx);
void deviation_update(struct DeviationPolicy *policy, struct Vector *curr, uint16_t precision);
void deviation_reset(struct DeviationPolicy *policy);

// Gambler Policy Operations
void gambler_policy_init(struct GamblerPolicy *policy, uint16_t *collectIndices[NUM_INDICES], FixedPoint predFeatures[], uint16_t numIndices, FixedPoint collectionRate, FixedPoint budget, FixedPoint alpha, FixedPoint beta, struct Vector *mean, struct Vector *dev);
uint8_t gambler_should_collect(struct GamblerPolicy *policy, uint16_t seqIdx);
void gambler_update(struct GamblerPolicy *policy, struct Vector *curr, uint16_t precision);
void gambler_update_policy(struct GamblerPolicy *policy);
void gambler_reset(struct GamblerPolicy *policy);

#endif
