## Makefile per la compilazione dei programmi di esempio del corso di
## High Performance Computing 2023/2024, corso di laurea in Ingegneria
## e Scienze Informatiche, Universita' di Bologna.
##
## Ultima modifica: 2023-09-09, Moreno Marzolla <moreno.marzolla(at)unibo.it>
##
## `make` senza parametri genera tutti i target
## `make help` per mostrare i target disponibili

EXE_OMP:=$(basename $(filter-out omp-bug1.c omp-bug2.c, $(wildcard omp-*.c)))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_SERIAL:=trap odd-even prefix-sum matmul-plain matmul-transpose game-of-life reduction vla-demo game-of-life-new
EXE_SIMD:=$(basename $(wildcard simd-*.c))
EXE_CUDA:=$(basename $(wildcard *.cu))
EXE_OPENCL:=$(basename $(wildcard opencl-*.c))
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_SIMD) $(EXE_OPENCL) $(EXE_CUDA)
DISTFILES:=$(wildcard *.cu *.c *.h *.cl *.sh *.ods) Makefile README cat-1344.pgm
DISTNAME:=HPC-2023-2024
CFLAGS+=-std=c99 -Wall -Wpedantic
MPICC?=mpicc
NVCC?=nvcc
MPICC?=mpicc
NVCFLAGS+=

all: mpi openmp serial simd opencl cuda

help:
	@echo
	@echo "Target disponibili:"
	@echo
	@echo "     help    stampa questo messaggio di aiuto"
	@echo "      all    compila tutto"
	@echo "   serial    compila tutti i programmi seriali"
	@echo "   openmp    compila tutti i programim OpenMP"
	@echo "      mpi    compila tutti i programmi MPI; richiede mpicc"
	@echo "   opencl    compila tutti i programmi OpenCL"
	@echo "     cuda    compila tutti i programmi CUDA"
	@echo "     simd    compila tutti i programmi SIMD"
	@echo "    clean    rimuove gli eseguibili e alcuni file temporanei"
	@echo
	@echo "Per installare MPI su Debian/Ubuntu:"
	@echo
	@echo "    sudo apt install mpi-default-bin mpi-default-dev"
	@echo
	@echo "Per installare OpenCL su Debian/Ubuntu:"
	@echo
	@echo "    sudo apt install pocl-opencl-icd    [per CPU]"
	@echo "    sudo apt install beignet-opencl-icd [per GPU Intel HD]"
	@echo "    sudo apt install libclc-amdgcn      [per GPU AMD]"
	@echo "    sudo apt install nvidia-opencl-icd  [per GPU Nvidia]"
	@echo
	@echo "Per scaricare CUDA SDK per la propria piattaforma:"
	@echo
	@echo "    https://developer.nvidia.com/cuda-downloads"
	@echo

dist: $(DISTNAME).zip

$(DISTNAME).zip: $(DISTFILES)
	rm -r -f $(DISTNAME)
	mkdir $(DISTNAME)
	cd $(DISTNAME)
	ln $(DISTFILES) $(DISTNAME)
	zip -r $(DISTNAME).zip $(DISTNAME)

gfx.o: gfx.c gfx.h

omp-mandelbrot: omp-mandelbrot.o gfx.o
omp-mandelbrot: LDLIBS+=-lX11
omp-mandelbrot: LDFLAGS+=-fopenmp

matmul-plain: LDLIBS+=-lrt
matmul-plain: matmul.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

matmul-transpose: CFLAGS+=-DTRANSPOSE
matmul-transpose: matmul.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

$(EXE_OMP): CFLAGS+=-fopenmp
$(EXE_OMP): LDLIBS+=-lrt
openmp: $(EXE_OMP)

$(EXE_MPI): CC=$(MPICC)
mpi: $(EXE_MPI)

mpi-vecsum mpi-vecsum2: LDLIBS+=-lm

serial: $(EXE_SERIAL)

$(EXE_SIMD): CFLAGS += -march=native
simd: $(EXE_SIMD)

simd-vsum-vector: CFLAGS+=-lm

simd-vsum-intrinsics: CFLAGS+=-lm

simd-vsum-auto: CFLAGS+=-lm -O2 -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized

simd-vsum-auto.s: simd-vsum-auto.c
	gcc -S -c -fverbose-asm -g -O2 -march=native -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized $< -o $@

simd-vsum-auto.lst: simd-vsum-auto.s
	as -alhnd $< > $@

$(EXE_OPENCL): CFLAGS+=simpleCL.c
$(EXE_OPENCL): LDLIBS+=-lOpenCL
opencl: $(EXE_OPENCL)

opencl-rotate opencl-matmul: LDLIBS+=-lm

cuda: $(EXE_CUDA)

cuda-rotate cuda-matmul: NVCFLAGS+=-lm

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

clean:
	\rm -f $(EXE) *.o *~ *.pbm *.s *.lst

distclean: clean
	\rm -r -f $(DISTNAME) $(DISTNAME).zip
