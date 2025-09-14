DEBUG ?= False
ENGINE ?= .

# Source Path.
LIBDIR=./lib/
SRCDIR=./src/
INCDIR=./include

# Source files.
SRC    = $(SRCDIR)mmo_math.c $(SRCDIR)mmo_file.c
SRCCPP = $(SRCDIR)retqss_pic.cc $(SRCDIR)retqss_boids.cc $(SRCDIR)retqss_md.cc $(SRCDIR)retqss_indinf.cc $(SRCDIR)retqss_covid19.cc $(SRCDIR)retqss_test.cc $(SRCDIR)retqss_social_force_model.cc $(SRCDIR)retqss_pathways.cc $(SRCDIR)retqss_utils.cc $(SRCDIR)retqss_subway_hub.cc $(SRCDIR)retqss_subway_stations.cc $(SRCDIR)retqss_classrooms.cc

TARGET = $(LIBDIR)libmmo_math.a $(LIBDIR)libmmo_file.a $(LIBDIR)libpic.a $(LIBDIR)libboids.a $(LIBDIR)libmd.a $(LIBDIR)libindinf.a $(LIBDIR)libcovid19.a $(LIBDIR)libtest.a $(LIBDIR)libsocial_force_model.a $(LIBDIR)libpathways.a $(LIBDIR)libutils.a $(LIBDIR)libsubway_hub.a $(LIBDIR)libsubway_stations.a $(LIBDIR)libclassrooms.a

RMS = rm -rf

OBJ = $(SRC:.c=.o)
OBJCC = $(SRCCPP:.cc=.o)

CFLAGS = -Wall -I$(INCDIR) -I$(ENGINE)/common -I$(ENGINE) -I$(ENGINE)/usr/include -lm -msse2 -mfpmath=sse -O2 -DNDEBUG

ifeq ($(DEBUG),True)
	CFLAGS += -g   
endif

CC=gcc

.SUFFIXES: .c .cc

default: $(TARGET) 

%.o: %.cc
	$(CXX) -std=c++11 -c $< -o $@ $(CFLAGS)

%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)


$(TARGET): $(OBJ) $(OBJCC)
	@mkdir -p $(LIBDIR)
	ar rsc $@ $(OBJ) $(OBJCC)  

clean:
	$(RMS) $(OBJ) $(OBJCC) $(TARGET)  

help:
	echo "make DEBUG=<True|False> ENGINE=<qss-engine-root-dir>"
	echo "Default values:"
	echo ""
	echo "ENGINE=."
