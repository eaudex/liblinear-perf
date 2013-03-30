CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -std=gnu++11
LIBS = blas/blas.a
SHVER = 1
OS = $(shell uname)
#LIBS = -lblas

all: train predict train-perf predict-perf bagging knn

lib: linear.o tron.o blas/blas.a
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,liblinear.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)

train: tron.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o $(LIBS)

predict: tron.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c tron.o linear.o $(LIBS)

train-perf: tron.o linear.o train-perf.cpp blas/blas.a eval.o common.o
	$(CXX) $(CFLAGS) -o train-perf train-perf.cpp tron.o linear.o eval.o common.o $(LIBS)

predict-perf: tron.o linear.o predict-perf.cpp blas/blas.a eval.o
	$(CXX) $(CFLAGS) -o predict-perf predict-perf.cpp tron.o linear.o eval.o $(LIBS)

bagging: tron.o linear.o bagging.cpp blas/blas.a eval.o common.o
	$(CXX) $(CFLAGS) -o bagging bagging.cpp tron.o linear.o eval.o common.o $(LIBS)

knn: knn.cpp eval.o common.o linear.o tron.o blas/blas.a
	$(CXX) $(CFLAGS) -o knn knn.cpp eval.o common.o linear.o tron.o $(LIBS)

eval.o: eval.cpp eval.h linear.h
	$(CXX) $(CFLAGS) -c -o eval.o eval.cpp

common.o: common.cpp
	$(CXX) $(CFLAGS) -c -o common.o common.cpp

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ tron.o linear.o train predict liblinear.so.$(SHVER)
	rm -f eval.o common.o train-perf predict-perf bagging knn

