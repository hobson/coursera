JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	DynamicClient.java \
	QuickFindUF.java \
	QuickFindClient.java \
	QuickUnionUF.java \
	QuickUnionClient.java \
	WeightedQuickUnionUF.java \
	WeightedClient.java \
	CompressedQuickUnionUF.java \
	CompressedClient.java

default: classes

classes: $(CLASSES:.java=.class)

all: classes

clean:
	$(RM) *.class