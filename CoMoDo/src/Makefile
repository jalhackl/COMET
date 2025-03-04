# --------------------------------------------------------------#
osv = Linux2.4
CFLAGS =  -O -pedantic -W -Wall 
CC = g++ -pedantic -W -Wall 
FC = g77  -W -Wall 
FFLAGS =  -O2 -W -Wall
# for profiling
#CC = cc -pedantic -W -Wall -pg
# ---------------------------------------------
LIBRARY  =  -Xlinker -defsym -Xlinker MAIN__=main 
# ---------------------------------------------
AR = ar rv
# ---------------------------------------------
CLUSTERER 	= main-DomainClusterer.o Network.o 
# ---------------------------------------------
TESTER          = DomainTester.o
#----------------------------------------------
all:	DomainClusterer DomainTester
new:	
	makedepend -fMakefile main-DomainClusterer.o
#	
DomainClusterer:	$(CLUSTERER)
	$(CC) $(CFLAGS) $(CLUSTERER) -o DomainClusterer

DomainTester:	$(TESTER)
	$(CC) $(CFLAGS) $(TESTER) -o DomainTester

clean:	
	rm -f *.o *~ *.a
# ---------------------------------------------
# DO NOT DELETE
