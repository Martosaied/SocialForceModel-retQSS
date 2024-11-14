# Target OS variables
ifneq ($(OS), Windows_NT)
OS = $(shell uname)
endif
DEBUG ?= False
CORES ?= 1
ARCH = $(shell uname -m)
ACCESS ?= HTTPS
ifeq ($(ACCESS), HTTPS)
ACCESS_PATH = https://github.com/
else
ACCESS_PATH = git@github.com:
endif

# The Directories, Source, Includes, Objects, Binary 
ROOT   	:= ..

# Repositories location.

mmo-compiler-repo 			:= ./mmoc 
mmo-interfaces-repo 		:= ./interfaces/sbml
mmo-models-repo 			:= $(ROOT)/models
mmo-packages-repo 			:= $(ROOT)/packages
qss-engine-repo 			:= ./engine 
qss-test-suite-repo			:= ./testsuite 
qss-solver-gui-repo			:= ./gui 
qss-user-libs-repo 			:= ./usr

all: mmo-compiler mmo-interfaces qss-engine qss-solver-gui qss-user-libs

mmo-interfaces: create-dirs 
	@echo BUILDING MMO-SBML INTERFACE
	@cd interfaces/sbml &&  $(MAKE) DEBUG=$(DEBUG) -j $(CORES)
	@echo Done

qss-engine: create-dirs  
	@echo BUILDING QSS LIBRARY
	@cd engine && $(MAKE) -j $(CORES)
	@echo Done
	@echo BUILDING QSS DEBUG LIBRARY
	@cd engine && $(MAKE) DEBUG=True RETQSS_DEBUG=True -j $(CORES)
	@echo "Installing engine libraries."
	ln -sf $(ROOT)/src/engine/usr/lib/* $(ROOT)/lib/
	@echo Done

qss-user-libs: create-dirs qss-engine
	@echo BUILDING USER LIBS
	@cd usr && $(MAKE) DEBUG=$(DEBUG) -j $(CORES) ENGINE=../engine     
	@echo Done

mmo-compiler: create-dirs 
	@echo BUILDING MICROMODELICA C COMPILER
	@cd mmoc && $(MAKE) DEBUG=$(DEBUG) -j $(CORES) 
	@echo Done

qss-solver-gui: create-dirs
	@echo BUILDING MMO EDITOR
	@cd gui && $(MAKE) -f gui.makefile      
	@echo Done

qss-solver-doc: qss-engine mmo-compiler qss-solver-gui mmo-interfaces
	@echo BUILDING PROJECT DOCUMENTATION
	@cd engine && $(MAKE) doc 
	@cd mmoc && $(MAKE) doc 
	@cd gui && $(MAKE) -f gui.makefile doc 
	@cd interfaces/sbml && $(MAKE) doc 
	
create-dirs:
	@mkdir -p $(ROOT)/bin
	@mkdir -p $(ROOT)/build
	@mkdir -p $(ROOT)/output
	@mkdir -p $(ROOT)/lib

install: all	
	@echo "Installing engine libraries."
	ln -sf $(ROOT)/src/engine/usr/lib/* $(ROOT)/lib/
	@echo "Installing MicroModelica Compiler."
	ln -sf $(ROOT)/src/mmoc/usr/bin/mmoc $(ROOT)/bin/
	@echo "Installing SBML translator."
	ln -sf $(ROOT)/src/interfaces/sbml/usr/bin/translate-sbml $(ROOT)/bin/
	@echo "Installing solver GUI."
	@install $(ROOT)/src/gui/usr/bin/qss-solver $(ROOT)/bin/
	@echo "Installing user libraries."
	ln -sf $(ROOT)/src/usr/lib/* $(ROOT)/lib/
ifeq ($(OS),Linux)
	@install $(ROOT)/deploy/linux/scripts/* $(ROOT)/bin/
	@install $(ROOT)/deploy/linux/qss-solver.ini.in $(ROOT)/bin/qss-solver.ini
	@install $(ROOT)/src/engine/3rd-party/partitioners/hmetis/khmetis $(ROOT)/bin/
ifeq ($(ARCH),i686)
	@install $(ROOT)/src/engine/3rd-party/partitioners/patoh/Linux-i386/libpatoh.a $(ROOT)/lib/
	@install $(ROOT)/src/engine/3rd-party/partitioners/metis/Linux-i386/libmetis.a $(ROOT)/lib/
else
	@install $(ROOT)/src/engine/3rd-party/partitioners/patoh/Linux-x86_64/libpatoh.a $(ROOT)/lib/
	@install $(ROOT)/src/engine/3rd-party/partitioners/metis/Linux-x86_64/libmetis.a $(ROOT)/lib/
endif
else
ifeq ($(OS),Windows_NT)
	@cp $(ROOT)/deploy/windows/scripts/* $(ROOT)/bin/
	@cp $(ROOT)/deploy/windows/qss-solver.ini.in $(ROOT)/bin/qss-solver.ini
else
	@cp $(ROOT)/deploy/osx/scripts/* $(ROOT)/bin/
	@cp $(ROOT)/deploy/osx/qss-solver.ini.in $(ROOT)/bin/qss-solver.ini
endif
endif
	
.PHONY: clean all 

clean: 
	@cd mmoc && $(MAKE) clean 
	@cd engine && $(MAKE) clean 
	@cd engine && $(MAKE) DEBUG=True clean 
	@cd usr && $(MAKE) clean
	@cd gui && $(MAKE) clean    
	@cd interfaces/sbml && $(MAKE) clean     

help:
	@echo "make DEBUG=<True|False> CORES=<n> ACCESS=<HTTPS|SSH>"
	@echo "Default values:"
	@echo ""
	@echo "DEBUG=False"
	@echo "CORES=1"
	@echo "ACCESS=HTTPS"
