
//import java.io.IOException;
//import java.io.InputStreamReader;

// "Dynamic Connectivity Client" from Week 1
// From 15UnionFind.pdf slide 10
class DynamicClient {
    public static void main(String[] args) {
    	int N = StdIn.readInt();
    	UF uf = new UF(N);
    	while (!StdIn.isEmpty()) {
    		int p = StdIn.readInt();
    		int q = StdIn.readInt();
    		if (!uf.connected(p, q)) {
    			uf.union(p, q);
    			StdOut.println(p + " " + q);
    		}
    	}
    } 
} // class UnionFindApp