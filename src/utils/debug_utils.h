// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef TRACE_MODE 
  #define TRACE_PRINTF(fmt, args...) fprintf(stderr, "TRACE: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
  #define TRACE_PRINTF(fmt, args...)
#endif

static void hexDump(const char * desc, const void * addr, const int len) {
	int i;
	unsigned char buff[17];
	const unsigned char * pc = (const unsigned char *)addr;

	// Output description if given.
	if (desc != NULL)
		printf ("%s:\n", desc);

	// Length checks.
	if (len == 0) {
		printf("  ZERO LENGTH\n");
		return;
	}
	else if (len < 0) {
		printf("  NEGATIVE LENGTH: %d\n", len);
		return;
	}

	// Process every byte in the data.
	for (i = 0; i < len; i++) {
		// Multiple of 16 means new line (with line offset).
		if ((i % 16) == 0) {
			// Don't print ASCII buffer for the "zeroth" line.
			if (i != 0)
				printf ("  %s\n", buff);

			// Output the offset.
			printf ("  %04x ", i);
		}

		// Now the hex code for the specific character.
		printf (" %02x", pc[i]);
		// And buffer a printable ASCII character for later.
		if ((pc[i] < 0x20) || (pc[i] > 0x7e)) // isprint() may be better.
			buff[i % 16] = '.';
		else
			buff[i % 16] = pc[i];
		buff[(i % 16) + 1] = '\0';
	}

	// Pad out last line if not exactly 16 characters.
	while ((i % 16) != 0) {
		printf ("   ");
		i++;
	}

	// And print the final ASCII buffer.
	printf ("  %s\n", buff);
}
