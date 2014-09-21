// fast Union using the lazy approach (but slow find)
// initialize=T(N), union=T(lg(N)), find=T(lg(N)) (because trees are very flat)
public class CompressedQuickUnionUF extends WeightedQuickUnionUF {

    public CompressedQuickUnionUF(int N) {
        super(N);
        __class__ = "CompressedQuickUnionUF";
    }

    // follow parent pointers until you find the base
    protected int root(int i) {
        while (i != id[i]) {
            id[i] = id[id[i]]; // while you're here take the parent's ID!
            i = id[i]; // why do this extra lookup/assignment?
        }
        return i; // why not just return id[i]? because one extra array lookup when i == id[i]
    }

    // now only need to check to see if they have the same tree root/base
    public boolean connected(int p, int q) {
        return root(p) == root(q);
    }

    // just change the nase of p (dest node) to point to the base of q (src node)
    public void union(int dest, int src) {
        int i = root(dest);
        int j = root(src);
        if (i == j)
            return;
        if (depth[i] < depth[j]) {
            id[i] = j;
            depth[j] += depth[i]; }
        else {
            id[j] = i;
            depth[i] += depth[j]; }
        id[i] = j;
    } // union(dest, src)

} // QuickUnionUF
