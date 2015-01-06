JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	DynamicClient.java \
	QuickFindUnion.java \
	QuickFindClient.java \
	QuickUnionFind.java \
	QuickUnionClient.java \
	WeightedQuickUnionFind.java \
	WeightedClient.java \
	CompressedQuickUnionFind.java \
	CompressedClient.java \
	Percolation.java \
	PercolationStats.java \
	PercolationTest.java

default: classes

classes: $(CLASSES:.java=.class)

all: classes

clean:
	$(RM) *.class