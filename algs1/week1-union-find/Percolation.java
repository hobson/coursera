public class Percolation {
    private int N = 8;
    private WeightedQuickUnionUF uf;
    private double maxN = Math.floor(Math.sqrt(1.0e9));
    private boolean wrap = false;
    private boolean[] openList;

    // create N-by-N grid, with all sites blocked
    public Percolation(int i) {
        // assume that machine has 16 GB of RAM and each site requires 16 bytes of storage
        if (i <= 0)  // || (i > maxN))
            throw new IllegalArgumentException(
                "Requested width of percolation matrix, " + i + ", is outside the range 1 to " + maxN);
        N = i;
        uf = new WeightedQuickUnionUF(N * N + 2);

        // connect the bottom row to the output node
        for (int j = 0; j < N; j++)
            uf.union(N * N - j, N * N + 1);

        openList = new boolean[N * N];
        for (int j = 0; j < N * N; j++)
            openList[j] = false;
        // node 0 is the "top" or input node that is connected to the input surface
        // node N+1 is the "bottom" or output node that is connected to the output surface
        // node 1 is the upper left node 
        // node N is the upper right node
        // node N+1 is the left-hand node of the second row
    }

    private boolean checkBounds(int i) {
        if (i <= 0 || i > N)
            throw new IndexOutOfBoundsException("Index " + i + " is outside the range 1 to " + N);
        else
            return true;
    }

    // open site (row i, column j) if it is not already
    public void open(int i, int j) {
        checkBounds(i);
        checkBounds(j);
        // StdOut.println("Connecting...");

        openList[(i - 1) * N + j -1] = true;

        if ((i == 1) || (isOpen(i - 1, j))) {
            // connect site above to this site
            uf.union(Math.max((i - 2) * N + j, 0), (i - 1) * N + j);
            // StdOut.println("  " + ((i-2)*N + j) + "<->" + ((i-1)*N + j));            
        }

        // connect this site to the site to the right, but only if it's open
        if (isOpen(i, ((j + 0) % N) + 1) && (wrap || (j < N))) {
            uf.union((i - 1) * N + j, (i - 1) * N + (j % N) + 1);
            // StdOut.println("  " + ((i-1)*N + j) + "<->" + ((i-1)*N + (j + 0) % N + 1));
        }

        // connect this site to the site to the left, but only if it's open
        if (j > 1) {
            if (isOpen(i, (j - 2) + 1)) {
                uf.union((i - 1) * N + j, (i - 1) * N + j - 1);
                // StdOut.println("  " + ((i-1)*N + j) + "<->" + ((i-1)*N + (j - 2) % N + 1));
            }
        }
        else if (wrap) {
            if (isOpen(i, N)) {
                uf.union((i - 1) * N + j, i * N + j - 1);
                // StdOut.println("  " + ((i-1)*N + j) + "<->" + ((i-1)*N + (j - 2) % N + 1));
            }
        }
    }

    // is site (row i, column j) open or connected?
    public boolean isOpen(int i, int j) {
        checkBounds(i);
        checkBounds(j);
        return openList[(i - 1) * N + j - 1];
    }

    // is site (row i, column j) full?
    public boolean isFull(int i, int j) {
        return (isOpen(i, j) && uf.connected(0, (i - 1)*N + j));
    }

    // does the system percolate?
    public boolean percolates() {
        return uf.connected(0, N*N+1);
    }

    // // test client, optional
    // private void check_one_case(String[] args) {
    //     N = 8;
    //     int attempts = 0;
    //     if (args.length > 0) {
    //         N = Integer.parseInt(args[0]);
    //     }
    //     Percolation p = new Percolation(N);
    //     for (int i=1; i<args.length; i++) {
    //         int num_holes = Integer.parseInt(args[i]);
    //         for (int j=0; j<num_holes; j++) {
    //             attempts = 0;
    //             while ((!random_hole()) && (attempts < max_attempts) && (j < num_holes)) {
    //                 attempts += 1;
    //                 // StdOut.println("attempt " + j + "." + attempts);
    //                 }
    //             // StdOut.println(j);
    //         }
    //         StdOut.println("With " + num_holes + " holes in a " + N + "x" + N + " matrix, percolates ?= " + p.percolates());
    //     }
    // }



}