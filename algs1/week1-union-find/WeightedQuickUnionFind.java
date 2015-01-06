// fast Union using the lazy approach (but slow find)
// initialize=T(N), union=T(lg(N)), find=T(lg(N)) (because trees are very flat)
public class WeightedQuickUnionFind extends QuickUnionFind {
    protected int[] depth;

    public WeightedQuickUnionFind(int N) {
        super(N);
        __class__ = "WeightedQuickUnionFind";
        depth = new int[N];
        for (int i = 0; i < N; i++) {
            depth[i] = 1;
        }
    }

    // now only need to check to see if they have the same tree root/base
    public boolean connected(int p, int q) {
        return root(p) == root(q);
    }

    // just change the nase of p (dest node) to point to the base of q (src node)
    public void union(int dest, int src) {
        int i = root(dest);
        int j = root(src);
        if (i == j) return;
        if (depth[i] < depth[j]) {  // null pointer exception
            id[i] = j;
            depth[j] += depth[i]; }
        else {
            id[j] = i;
            depth[i] += depth[j]; }
    } // union(dest, src)

    public String str() {
        String s = super.str();
        s += '\n';
        for (int i=0; i<id.length-1; i++) {
            s += depth[i] + " ";
        } // for i in range(N)
        s += depth[id.length-1];
        return s;
    } // str()
} // QuickUnionUF
