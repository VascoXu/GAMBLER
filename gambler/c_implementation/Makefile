CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: policy

policy: utils/constants.h utils/fixed_point.h data.h sampler.h utils/bitmap.h utils/encoding.h utils/matrix.h policy_parameters.h policy.h main.h decision_tree.h parameters.h utils/fixed_point.c utils/bitmap.c utils/matrix.c policy.c main.c decision_tree.c
	$(CC) -o policy utils/fixed_point.c sampler.c utils/bitmap.c utils/encoding.c utils/matrix.c  policy.c main.c decision_tree.c

inference_tests: tests/inference_tests.h policy_parameters.h tests/data.h decision_tree.h tests/parameters.h tests/inference_tests.c
	$(CC) -o inference_tests decision_tree.c tests/inference_tests.c

clean: 
	rm -f *.o policy inference_tests *.dSYM
