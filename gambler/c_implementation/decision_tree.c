#include "decision_tree.h"


uint8_t decision_tree_inference(int16_t *inputs, struct decision_tree *tree) {
    // Start inference at the root node
    volatile uint8_t treeIdx = 0;
    volatile int16_t featureIdx;
    volatile int16_t threshold;

    while ((tree->leftChildren[treeIdx] >= 0) && (tree->rightChildren[treeIdx] >= 0)) {
	    threshold = tree->thresholds[treeIdx];
	    featureIdx = tree->features[treeIdx];

	    if (inputs[featureIdx] <= threshold) {
	        treeIdx = tree->leftChildren[treeIdx];
	    } else {
	        treeIdx = tree->rightChildren[treeIdx];
	    }
    }

    return tree->predictions[treeIdx];
}