INCLUDE_PATH		= -I./include 
LIBRARY_PATH		=

TARGET = cnn
CC = g++
LD = g++
CFLAGS = -O3 -fopenmp $(INCLUDE_PATH) 
LFLAGS = -O3 -fopenmp $(LIBRARY_PATH)
LIBS = -larmadillo

########################################################################################
## !! Do not edit below this line

HEADERS := $(wildcard include/*.h)
SOURCES := $(wildcard src/*.cpp)

OBJECTS := $(addprefix obj/,$(notdir $(SOURCES:.cpp=.o)))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LD) $(OBJECTS) -o $(TARGET) $(CFLAGS) $(LFLAGS) $(LIBS)

obj/%.o: src/%.cpp | obj
	$(CC) -c $< -o $@ $(CFLAGS) 

obj:
	mkdir obj

clean:
	rm -f $(OBJECTS)
	rm -f $(TARGET)

