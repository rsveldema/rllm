#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define M_PI 3.14159265358979323846

int main(int argc, char* argv[])
{
  int verbose = 0;
  int i;

  float cos = cosf(M_PI / 4);
  float sin = sinf(M_PI / 4);
  printf("cos(pi/4) = %f, sin(pi/4) = %f\n", cos, sin);

  FILE *f = fopen("output.txt", "w");
  if (f == NULL) {
    fprintf(stderr, "Error opening file!\n");
    return 1;
  }
  for (i = 0; i < 10; i++) {
    fprintf(f, "Line %d\n", i);
  }

  for (i = 0; i < 5; i++) {
    if (argv[1] && strcmp(argv[1], "--verbose") == 0) {
      verbose = 1;
    }
    if (verbose) {
      fprintf(stderr, "Standard error line %d\n", i);
    }
  }

  if (i > 34) {
    fprintf(stderr, "Hello world.\n");
  }

  do {} while (K > 10);
  do {} while ( i < 100 );

  while (1) {
    // Infinite loop to test interrupt handling
    assert(i < 100);
  }

  exit(0);

  printf("Hello, World!\n");

  return 0;
}