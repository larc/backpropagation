INCLUDE_PATH		= -I./include 
LIBRARY_PATH		=
BLAS_LIBS			= -lumfpack
SUITESPARSE_LIBS	= -lspqr -lcholmod

TARGET = cnn
CC = g++
LD = g++
CFLAGS = -O3 -fopenmp $(INCLUDE_PATH) 
LFLAGS = -O3 -fopenmp $(LIBRARY_PATH) -lX11 -lpthread
LIBS = $(OPENGL_LIBS) $(SUITESPARSE_LIBS) $(BLAS_LIBS) -larmadillo

########################################################################################
## !! Do not edit below this line

HEADERS := $(wildcard include/*.h)
SOURCES := $(wildcard src/*.cpp) $(wildcard src/viewer/*.cpp)

OBJECTS := $(addprefix obj/,$(notdir $(SOURCES:.cpp=.o)))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LD) $(OBJECTS) -o $(TARGET) $(CFLAGS) $(LFLAGS) $(LIBS)

obj/%.o: src/%.cpp
	$(CC) -c $< -o $@ $(CFLAGS) 

clean:
	rm -f $(OBJECTS)
	rm -f $(TARGET)

