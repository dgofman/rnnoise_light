# Makefile to build rnnoise_light

CC = gcc
CFLAGS = -g -O2 -pedantic -Wall -Wextra -Wno-sign-compare \
  -Wno-parentheses -Wno-long-long -fvisibility=hidden \
  -DDLL_EXPORT -DPIC -I. -I./include -I./src

OBJDIR = .libs
SRC_OBJS = $(OBJDIR)/denoise.o $(OBJDIR)/rnn.o $(OBJDIR)/pitch.o \
           $(OBJDIR)/kiss_fft.o $(OBJDIR)/nnet.o $(OBJDIR)/rnnoise_tables.o
SRC_PATHS = src/denoise.c src/rnn.c src/pitch.c src/kiss_fft.c src/nnet.c src/rnnoise_tables.c

LIB_NAME = $(OBJDIR)/msys-rnnoise-0.dll
LIB_IMPLIB = $(OBJDIR)/librnnoise.dll.a

BIN = examples/rnnoise_demo.exe
BIN_SRC = examples/rnnoise_demo.c

.PHONY: all clean $(OBJDIR)

all: clean $(BIN)

clean:
	rm -rf $(OBJDIR)
	rm -f examples/*.exe

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile each source file to .libs/*.o
$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build shared DLL from .libs/*.o
$(LIB_NAME): $(SRC_OBJS)
	$(CC) -g -O2 -shared $^ \
	-Wl,--enable-auto-image-base \
	-Wl,--out-implib,$(LIB_IMPLIB) \
	-o $(LIB_NAME)

# Link final executable
$(BIN): $(BIN_SRC) $(LIB_NAME)
	$(CC) -g -o $@ $< $(LIB_IMPLIB)
