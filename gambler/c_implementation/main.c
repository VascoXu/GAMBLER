#include "main.h"

static FixedPoint DATA_BUFFER[SEQ_LENGTH * NUM_FEATURES];
static struct Vector featureVectors[SEQ_LENGTH];

static FixedPoint ZERO_BUFFER[NUM_FEATURES];
static struct Vector ZERO_FEATURES = { ZERO_BUFFER, NUM_FEATURES };


int main(void) {
    char *feature;
    FixedPoint featureVal;

    // Create the data buffer
    uint16_t i;
    for (i = 0; i < SEQ_LENGTH; i++) {
        featureVectors[i].data = DATA_BUFFER + (NUM_FEATURES * i);
        featureVectors[i].size = NUM_FEATURES;
    }

    for (i = 0; i < NUM_FEATURES; i++) {
        ZERO_BUFFER[i] = 0;
    }

    // Make the bitmap for this sequence
    uint16_t numBytes = (SEQ_LENGTH / BITS_PER_BYTE);
    if ((WINDOW_SIZE % BITS_PER_BYTE) > 0) {
        numBytes += 1;
    }

    uint8_t collectedBuffer[numBytes];
    struct BitMap collectedIndices = { collectedBuffer, numBytes };
    uint8_t outputBuffer[1024];

    // Make the policy
    #ifdef IS_UNIFORM
    struct UniformPolicy policy;
    uniform_policy_init(&policy, COLLECT_INDICES, UNIFORM_BUDGET);

    #elif defined(IS_ADAPTIVE_HEURISTIC)
    struct Vector *prevFeatures;
    
    struct HeuristicPolicy policy;
    heuristic_policy_init(&policy, MAX_SKIP, MIN_SKIP, THRESHOLD);
    
    #elif defined(IS_ADAPTIVE_DEVIATION)
    struct DeviationPolicy policy;

    FixedPoint meanData[NUM_FEATURES];
    struct Vector mean = { meanData, NUM_FEATURES };

    FixedPoint devData[NUM_FEATURES];
    struct Vector dev = { devData, NUM_FEATURES };

    deviation_policy_init(&policy, MAX_SKIP, MIN_SKIP, THRESHOLD, ALPHA, BETA, &mean, &dev);

    #elif defined(IS_ADAPTIVE_GAMBLER)
    struct GamblerPolicy policy;

    FixedPoint meanData[NUM_FEATURES];
    struct Vector mean = { meanData, NUM_FEATURES };

    FixedPoint devData[NUM_FEATURES];
    struct Vector dev = { devData, NUM_FEATURES };

    // Initialize values to zero (ps. sometimes they are initialized to random values?)
    for (i = 0; i < NUM_FEATURES; i++) {
        mean.data[i] = 0;
        dev.data[i] = 0;
    }

    FixedPoint PRED_FEATURES[NUM_TREE_FEATURES];

    gambler_policy_init(&policy, COLLECT_INDICES, PRED_FEATURES, NUM_INDICES, COLLECTION_RATE, BUDGET, ALPHA, BETA, &mean, &dev);
    #endif

    // Indices to sample data
    uint16_t seqIdx;
    uint16_t elemIdx;
    uint16_t windowIdx = 0;

    uint32_t collectCount = 0;
    uint32_t totalCount = 0;
    uint32_t count = 0;
    uint32_t idx = 0;

    uint8_t shouldCollect = 0;
    uint8_t didCollect = 1;

    for (seqIdx = 0; seqIdx < MAX_NUM_SEQ; seqIdx++) {
        // Clear the collected bit map
        clear_bitmap(&collectedIndices);

        #ifdef IS_UNIFORM
        uniform_reset(&policy);
        #elif defined(IS_ADAPTIVE_HEURISTIC)
        heuristic_reset(&policy);
        prevFeatures = &ZERO_FEATURES;
        #elif defined(IS_ADAPTIVE_DEVIATION)
        deviation_reset(&policy);
        #elif defined(IS_ADAPTIVE_GAMBLER)
        gambler_reset(&policy);
        #endif

        count = 0;

        // Iterate through the elements and select elements to keep.
        for (elemIdx = 0; elemIdx < SEQ_LENGTH; elemIdx++) {
            #ifdef IS_UNIFORM
            shouldCollect = uniform_should_collect(&policy, elemIdx);
            #elif defined(IS_ADAPTIVE_HEURISTIC)
            shouldCollect = heuristic_should_collect(&policy, elemIdx);
            #elif defined(IS_ADAPTIVE_DEVIATION)
            shouldCollect = deviation_should_collect(&policy, elemIdx);
            #elif defined(IS_ADAPTIVE_GAMBLER)
            shouldCollect = gambler_should_collect(&policy, windowIdx);
            #endif

            if (shouldCollect) {
                collectCount++;
                count++;

                // Collect the data
                didCollect = get_measurement((featureVectors + elemIdx)->data, seqIdx, elemIdx, NUM_FEATURES, SEQ_LENGTH);

                if (!didCollect) {
                    printf("ERROR. Could not collect data at Seq %d, Element %d\n", seqIdx, elemIdx);
                    break;
                }

                // Record the collection of this element.
                set_bit(elemIdx, &collectedIndices);

                #ifdef IS_ADAPTIVE_HEURISTIC
                heuristic_update(&policy, featureVectors + elemIdx, prevFeatures);
                prevFeatures = featureVectors + elemIdx;
                #elif defined(IS_ADAPTIVE_DEVIATION)
                deviation_update(&policy, featureVectors + elemIdx, DEFAULT_PRECISION);
                #elif defined(IS_ADAPTIVE_GAMBLER)
                gambler_update(&policy, featureVectors + elemIdx, DEFAULT_PRECISION);
                #endif
            }

            windowIdx++;

            if (windowIdx == WINDOW_SIZE) {
                #ifdef IS_ADAPTIVE_GAMBLER
                gambler_update_policy(&policy);
                gambler_reset(&policy);
                #endif

                windowIdx = 0;
            }

            totalCount++;
        }

        // Encode the collected elements
        encode_standard(outputBuffer, featureVectors, &collectedIndices, NUM_FEATURES, SEQ_LENGTH);

        // Print encoded elements
        #ifdef DEBUG
        int outputIdx;
        for (outputIdx = 0; outputIdx < 1024; outputIdx++) {
            printf("\\x%02x", outputBuffer[outputIdx]);
        }
        printf("\n");
        #endif

        if (!didCollect) {
            break;
        }
    }

    printf("\n");

    float rate = ((float) collectCount) / ((float) totalCount);
    printf("Collection Rate: %d / %d (%f)\n", collectCount, totalCount, rate);

    return 0;
}


void print_message(uint8_t *buffer, uint16_t numBytes) {
    uint16_t i;
    for (i = 0; i < numBytes; i++) {
        printf("\\x%02x", buffer[i]);
    }
    printf("\n");
    printf("Num Bytes: %d\n", numBytes);
}
