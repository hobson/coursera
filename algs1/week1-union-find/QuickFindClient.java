
//import java.io.IOException;
//import java.io.InputStreamReader;

// "Dynamic Connectivity Client" from Week 1
// From 15UnionFind.pdf slide 10


class QuickFindClient {
    public static int N = 100;

    public static void read_stdin(QuickFindUF uf) {
        StdOut.println(uf.str());
        while (!StdIn.isEmpty()) {
            int p = StdIn.readInt();
            int q = StdIn.readInt();
            StdOut.println(uf.str());
            StdOut.println(p + "<-" + q + " ");
            if (!uf.connected(p, q)) {
                uf.union(p, q);  // hangs when attempting to union 3-8, perhaps because depth=0?
            }
        }
        StdOut.println("Resultant array of " + N + " IDs (nodes) is:");
        StdOut.println(uf.str());
    }

    public static void main(String[] args) {
        N = StdIn.readInt();
        StdOut.println("Initializing an array of " + N + " IDs (nodes).");
        QuickFindUF uf = new QuickFindUF(N);
        read_stdin(uf);
    } 
} // class UnionFindApp