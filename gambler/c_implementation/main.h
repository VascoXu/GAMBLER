#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "policy.h"
#include "utils/bitmap.h"
#include "utils/matrix.h"
#include "policy_parameters.h"
#include "sampler.h"
#include "parameters.h"
#include "decision_tree.h"

#ifndef MAIN_H_
#define MAIN_H_

int main(void);
void print_message(uint8_t *buffer, uint16_t numBytes);

#endif
