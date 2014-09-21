// fast Union using the lazy approach (but slow find)
// initialize=T(N), union=T(N) (because have to find roots), find=O(N) (worst case for a really tall tree)
public class QuickUnionUF extends QuickFindUF {

    public QuickUnionUF(int N) {
        super(N);
    }

    // follow parent pointers until you find the root
    protected int root(int i) {
        while (i != id[i]) {
            i = id[i];
            }
        return i; 
    }

    // now only need to check to see if they have the same tree root/base
    public boolean connected(int p, int q) {
        return root(p) == root(q);
    } 

    // just change the ID of p (the destination node) to point to the base of q (the source node)
    public void union(int dest, int src) {
        int i = root(dest);
        int j = root(src);
        id[i] = j;
    } // union(p, q)
} // QuickUnionUF
