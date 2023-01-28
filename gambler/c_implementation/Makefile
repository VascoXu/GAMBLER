CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: policy

policy: utils/constants.h utils/fixed_point.h data.h sampler.h utils/bitmap.h utils/matrix.h policy_parameters.h policy.h main.h utils/fixed_point.c utils/bitmap.c utils/matrix.c policy.c main.c
	$(CC) -o policy utils/fixed_point.c sampler.c utils/bitmap.c utils/matrix.c  policy.c main.c

clean: 
	rm -f *.o policy *.dSYM
