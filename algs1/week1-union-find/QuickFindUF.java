
// Slow Dynamic Connectivity `union`, but fast `find`:
//    initialize T(N), union=T(N), find=T(1) 
//    So N union operations for N nodes (an average connectivity of 1) costs N^2 array accesses
public class QuickFindUF {
    protected String __class__ = "QuickFindUF";
	protected int[] id;

    // public QuickFindUF() {
    //     this(100);
    // }

    public QuickFindUF(int N) {
        // must know the size of the array/network before initializing it
        id = new int[N];
        // initialize the ID of all elements to it's index number
        // (each node is isolated in it's own "group" before `union` operations)
        for (int i = 0; i < N; i++)
            id[i] = i;
    }

    // check whether p and q are in the same group/union/component
    public boolean connected(int p, int q) {
        return id[p] == id[q];
    }

    public void union(int p, int q) {
        // set id of each object to itself
        int pid = id[p];
        int qid = id[q];
        // change all nodes to that have the ID that node p has to have the ID of node q
        for (int i = 0; i < id.length; i++) {
            if (id[i] == pid)
                id[i] = qid;
        }
    } // union(p,q)

    public String str() {
        String s = new String(__class__ + '\n');
        for (int i=0; i<id.length-1; i++) 
            s += i + " ";
        s += id.length-1;
        s += '\n';
        for (int i=0; i<id.length-1; i++)
            s += id[i] + " ";
        s += id[id.length-1];
        return s;
    } // str()
} // QuickFindUF
