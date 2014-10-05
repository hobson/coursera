
//import java.io.IOException;
//import java.io.InputStreamReader;

// "Dynamic Connectivity Client" from Week 1
// From 15UnionFind.pdf slide 10


class CompressedClient extends QuickFindClient {

    public static void main(String[] args) {
        N = StdIn.readInt();
        StdOut.println("Initializing an array of " + N + " IDs (nodes).");
        CompressedQuickUnionFind uf = new CompressedQuickUnionFind(N);
        read_stdin(uf);
    } 
}