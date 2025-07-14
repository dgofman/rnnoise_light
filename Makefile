EXAMPLE_DIR = examples
SRC_DEMO = $(EXAMPLE_DIR)/rnnoise_demo.c

# Output files
DEMO_EXE = $(EXAMPLE_DIR)/rnnoise_demo.exe
DLL_NAME = $(EXAMPLE_DIR)/rnnoise-light.dll
IMPLIB   = rnnoise-light.a

# Build flags
CC = gcc
CFLAGS = -O2 -Wall -Wextra -pedantic -march=native -fPIC \
         -Wconversion -Wshadow -Wfloat-equal -Wcast-align -Wundef \
         -Wduplicated-cond -Wduplicated-branches -Wlogical-op
#CFLAGS = -O2 -Wall -Wextra -march=native -fPIC
LDFLAGS = -shared -Wl,--out-implib,$(IMPLIB)

# Source and object files
SRC_FILES = \
	src/denoise.c \
	src/rnn.c \
	src/pitch.c \
	src/kiss_fft.c \
	src/nnet.c \
	src/rnnoise_data.c \
	src/rnnoise_tables.c

OBJECTS = $(SRC_FILES:.c=.o)

.PHONY: all 

all: clean $(DLL_NAME) $(DEMO_EXE)

clean:
	rm -rf $(OBJDIR) $(DEMO_EXE) $(DLL_NAME) $(IMPLIB) $(OBJECTS) $(EXAMPLE_DIR)/*.o

# Rule to build the DLL
$(DLL_NAME): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

# Rule to build the demo binary, linked against the DLL
$(DEMO_EXE): $(SRC_DEMO) $(IMPLIB)
	$(CC) $(CFLAGS) -o $@ $< $(IMPLIB)