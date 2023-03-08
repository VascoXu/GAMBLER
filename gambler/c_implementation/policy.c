#include "policy.h"

static FixedPoint POLICY_BUFFER[256];


/**
 * Uniform Policy Functions
 */
void uniform_policy_init(struct UniformPolicy *policy, uint16_t *collectIndices[NUM_INDICES], uint16_t numIndices) {
    policy->collectIndices = collectIndices;
    policy->numIndices = numIndices;
    policy->collectIdx = 0;
}


uint8_t uniform_should_collect(struct UniformPolicy *policy, uint16_t seqIdx) {
    if (policy->collectIdx >= policy->numIndices) {
        return 0;
    }

    uint8_t result = (seqIdx == policy->collectIndices[policy->numIndices-OFFSET][policy->collectIdx]);
    policy->collectIdx += result;
    return result;
}


//void uniform_update(struct UniformPolicy *policy) {
//    policy->collectIdx += 1;
//}


void uniform_reset(struct UniformPolicy *policy) {
    policy->collectIdx = 0;
}

/**
 * Heuristic Policy Functions
 */
void heuristic_policy_init(struct HeuristicPolicy *policy, uint16_t maxSkip, uint16_t minSkip, FixedPoint threshold) {
    policy->maxSkip = maxSkip;
    policy->minSkip = minSkip;
    policy->threshold = threshold;
    policy->currentSkip = 0;
    policy->sampleSkip = 0;
}

uint8_t heuristic_should_collect(struct HeuristicPolicy *policy, uint16_t seqIdx) {
    uint8_t result = (policy->sampleSkip == 0) || (seqIdx == 0);
    policy->sampleSkip -= 1;
    return result;
}


void heuristic_update(struct HeuristicPolicy *policy, struct Vector *curr, struct Vector *prev) {
    FixedPoint norm = vector_diff_norm(curr, prev);

    if (norm >= policy->threshold) {
        policy->currentSkip = policy->minSkip;
    } else {
        uint16_t nextSkip = policy->currentSkip + 1;
        uint8_t cond = nextSkip < policy->maxSkip;
        policy->currentSkip = cond * nextSkip + (1 - cond) * policy->maxSkip;
    }

    policy->sampleSkip = policy->currentSkip;
}


void heuristic_reset(struct HeuristicPolicy *policy)  {
    policy->sampleSkip = 0;
    policy->currentSkip = 0;
}


/**
 * Deviation Policy Functions
 */
void deviation_policy_init(struct DeviationPolicy *policy, uint16_t maxSkip, uint16_t minSkip, FixedPoint threshold, FixedPoint alpha, FixedPoint beta, struct Vector *mean, struct Vector *dev) {
    policy->maxSkip = maxSkip;
    policy->minSkip = minSkip;
    policy->currentSkip = 0;
    policy->sampleSkip = 0;
    policy->threshold = threshold;
    policy->alpha = alpha;
    policy->beta = beta;
    policy->mean = mean;
    policy->dev = dev;
}


uint8_t deviation_should_collect(struct DeviationPolicy *policy, uint16_t seqIdx) {
    uint8_t result = (policy->sampleSkip == 0) || (seqIdx == 0);
    policy->sampleSkip -= 1;
    return result;
}


void deviation_update(struct DeviationPolicy *policy, struct Vector *curr, uint16_t precision) {
    policy->mean = vector_gated_add_scalar(policy->mean, curr, policy->mean, policy->alpha, precision);

    struct Vector temp = { POLICY_BUFFER, curr->size };
    vector_absolute_diff(&temp, curr, policy->mean);

    policy->dev = vector_gated_add_scalar(policy->dev, &temp, policy->dev, policy->beta, precision);
    FixedPoint norm = vector_norm(policy->dev);

    printf("norm: %d\n", norm);

    if (norm >= policy->threshold) {
        uint16_t nextSkip = (policy->currentSkip) >> 1;
        uint8_t cond = nextSkip < policy->minSkip;
        policy->currentSkip = cond * policy->minSkip + (1 - cond) * nextSkip;
    } else {
        uint16_t nextSkip = policy->currentSkip + 1;
        uint8_t cond = nextSkip < policy->maxSkip;
        policy->currentSkip = cond * nextSkip + (1 - cond) * policy->maxSkip;
    }

    policy->sampleSkip = policy->currentSkip;
}


void deviation_reset(struct DeviationPolicy *policy)  {
    policy->sampleSkip = 0;
    policy->currentSkip = 0;
    vector_set(policy->mean, 0);
    vector_set(policy->dev, 0);
}

/**
 * Gambler Policy Functions
 */

void gambler_policy_init(struct GamblerPolicy *policy, uint16_t *collectIndices[NUM_INDICES], FixedPoint predFeatures[], uint16_t numIndices, FixedPoint collectionRate, FixedPoint budget, FixedPoint alpha, FixedPoint beta, struct Vector *mean, struct Vector *dev) {
    policy->collectIndices = collectIndices;
    policy->collectionRate = collectionRate;
    policy->budget = budget;
    policy->samplesLeft = TOTAL_SAMPLES;
    policy->numIndices = numIndices;
    policy->predFeatures = predFeatures;
    policy->collectIdx = 0;
    policy->collected = 0;
    policy->alpha = alpha;
    policy->beta = beta;
    policy->mean = mean;
    policy->dev = dev;
    policy->meanDev = 0;

    // TODO: mean and dev are not always initialized to 0?
}

uint8_t gambler_should_collect(struct GamblerPolicy *policy, uint16_t seqIdx) {
    if (policy->collectIdx >= policy->numIndices || policy->collected >= policy->budget) {
        return 0;
    }

    uint8_t result = (seqIdx == policy->collectIndices[policy->numIndices-OFFSET][policy->collectIdx]);
    policy->collectIdx += result;
    policy->collected += result;
    return result;
}

void gambler_update(struct GamblerPolicy *policy, struct Vector *curr, uint16_t precision) {
    int i; 
    policy->mean = vector_gated_add_scalar(policy->mean, curr, policy->mean, policy->alpha, precision);

    struct Vector temp = { POLICY_BUFFER, curr->size };
    vector_absolute_diff(&temp, curr, policy->mean);

    policy->dev = vector_gated_add_scalar(policy->dev, &temp, policy->dev, policy->beta, precision);
    FixedPoint norm = vector_norm(policy->dev);

    policy->meanDev += norm;
}

void gambler_update_policy(struct GamblerPolicy *policy) {
    policy->samplesLeft -= WINDOW_SIZE;

    uint32_t samplesLeft = policy->samplesLeft;
    uint32_t leftover = (policy->budget - policy->collected);
    uint32_t budgetLeft = ((leftover*WINDOW_SIZE) + (samplesLeft - 1))/samplesLeft; // round up

    // Predict collection rate using decision tree
    policy->meanDev /= WINDOW_SIZE;
    FixedPoint meanDev = policy->meanDev / WINDOW_SIZE;
    FixedPoint norm = vector_norm(policy->dev);
    policy->predFeatures = (FixedPoint []) {policy->meanDev, policy->collectionRate};
    uint8_t toCollect = decision_tree_inference(policy->predFeatures, &TREE);

    // Compute the weighted average of predicted and uniform collection rate
    uint32_t weight = fp32_div(INT_TO_FIXED(leftover), INT_TO_FIXED(policy->budget), DEFAULT_PRECISION);
    uint32_t collectionRate = FIXED_TO_INT(fp_gated_add_scalar(INT_TO_FIXED(toCollect), INT_TO_FIXED(budgetLeft), weight, DEFAULT_PRECISION));

    // Clip the collection rate
    if (collectionRate > WINDOW_SIZE) {
        collectionRate = WINDOW_SIZE;
    }

    policy->numIndices = collectionRate;
    policy->meanDev = 0;
}

void gambler_reset(struct GamblerPolicy *policy) {
    policy->collectIdx = 0;
}
