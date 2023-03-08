#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "policy.h"
#include "data.h"
#include "utils/bitmap.h"
#include "utils/encoding.h"
#include "utils/matrix.h"
#include "policy_parameters.h"
#include "sampler.h"
#include "decision_tree.h"
#include "parameters.h"

#ifndef MAIN_H_
#define MAIN_H_

int main(void);
void print_message(uint8_t *buffer, uint16_t numBytes);

#endif
