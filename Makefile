CC=nvcc
INCLUDE_FLAGS := -I./src
C_FLAGS_NODE3=--std=c++20 --extended-lambda $(INCLUDE_FLAGS)
DEBUG_FLAGS=-g -G

SRC=$(wildcard src/*.cpp src/*.cu) 
TARGET=compress

all: ${TARGET}

${TARGET}: ${SRC}
	${CC} ${C_FLAGS_NODE3} -o $@ $^

debug: ${TARGET}_debug

${TARGET}_debug: ${SRC}
	${CC} ${C_FLAGS_NODE3} ${DEBUG_FLAGS} -o $@ $^

clean:
	rm -f ${TARGET} ${TARGET}_debug

.PHONY: clean