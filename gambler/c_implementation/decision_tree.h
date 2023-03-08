#include <stdint.h>
#include "utils/fixed_point.h"


#ifndef DECISIONTREE_H_
#define DECISIONTREE_H_

struct decision_tree {
    uint16_t numNodes;
    int16_t *thresholds;
    int16_t *features;
    uint16_t *predictions;
    int16_t *leftChildren;
    int16_t *rightChildren;
};

uint8_t decision_tree_inference(int16_t *inputs, struct decision_tree *tree);

#endif
