#include "inference_tests.h"


int main(void) {
    int16_t inputFeatures[NUM_FEATURES];
    uint8_t featureIdx = 0;
    uint8_t label = 0;
    uint32_t isCorrect = 0;
    uint32_t totalCount = 0;
    
    uint8_t pred;
    uint16_t k;

    for (int k = 0; k < NUM_INPUTS; k++) {

        for (int j = 0; j < NUM_FEATURES; j++) {
            inputFeatures[j] = DATASET_INPUTS[k * NUM_FEATURES + j];
        }

        label = DATASET_LABELS[k];

	    pred = decision_tree_inference(inputFeatures, &TREE);
        printf("norm: %hd | rate: %hd | pred: %hhu | label:%d \n", inputFeatures[0], inputFeatures[1], pred, label);
	    isCorrect += (pred == label);
	    totalCount += 1;
    }

    printf("Accuracy: %d / %d\n", isCorrect, totalCount);
    return 0;
}
