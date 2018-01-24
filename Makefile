# makefile for real cuda codeGen

CC = nvcc
CC_OPTS = -Wno-deprecated-gpu-targets

TARGETS = realcuda

.PHONY: all build clean

.DEFAULT: build

all: build

build: $(TARGETS)

%: %.o fiddlelink.o fiddlelink.h
	$(CC) $(CC_OPTS) -o $@ $@.o fiddlelink.o

%.o: %.cu fiddlelink.h
	$(CC) $(CC_OPTS) -c $<

fiddlelink.o: fiddlelink.h

clean:
	rm -f *.o $(TARGETS) 
