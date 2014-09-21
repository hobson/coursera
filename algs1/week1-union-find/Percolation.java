import static java.lang.Math.*;

public class Percolation {
    public static int N = 8;
    public static CompressedQuickUnionUF uf;
    public static double maxN = floor(sqrt(1.0e9));
    private static double max_attempts = maxN;
    public static boolean wrap = true;

    // create N-by-N grid, with all sites blocked
    public Percolation(int i) throws IllegalArgumentException {
        // assume that machine has 16 GB of RAM and each site requires 16 bytes of storage
        if ((i <= 0) || (i > maxN))
            throw new IllegalArgumentException(
                "Requested width of percolation matrix, " + i + ", is outside the range 1 to " + maxN);
        N = i;
        uf = new CompressedQuickUnionUF(i * i + 2);
        for (i=0; i<N; i++)
            uf.union(N*N-i, N*N+1);

        // node 0 is the "top" or input node that is connected to the input surface
        // node N+1 is the "bottom" or output node that is connected to the output surface
        // node 1 is the upper left node 
        // node N is the upper right node
        // node N+1 is the left-hand node of the second row
    }

    public static boolean check_bounds(int i) throws IndexOutOfBoundsException {
        if (i < 0 || i > N)
            throw new IndexOutOfBoundsException("Index " + i + " is outside the range 1 to " + N);
        else
            return true;
    }

    // open site (row i, column j) if it is not already
    public static void open(int i, int j) {
        check_bounds(i);
        check_bounds(j);
        StdOut.println("Connecting...");
        if (i == 1) {
            uf.union(0, j);
            StdOut.println("  " + 0 + "<->" + j);
        }
        else {
            // connect site above to this site
            uf.union((i-2)*N + j, (i-1)*N + j);
            StdOut.println("  " + ((i-2)*N + j) + "<->" + ((i-1)*N + j));
        }
        // connect this site to the site to the right, but only if it's open
        if (isOpen(i, (j+0)%N+1) && (wrap || (j < N))) {
            uf.union((i-1)*N + j, (i-1)*N + (j + 0) % N + 1);
            StdOut.println("  " + ((i-1)*N + j) + "<->" + ((i-1)*N + (j + 0) % N + 1));
        }
        // connect this site to the site to the left, but only if it's open
        if (isOpen(i, (j-2)%N+1) && (wrap || (j > 1))) {
            uf.union((i-1)*N + j, (i-1)*N + (j - 2) % N + 1);
            StdOut.println("  " + ((i-1)*N + j) + "<->" + ((i-1)*N + (j - 2) % N + 1));
        }
    }

    // is site (row i, column j) open or connected?
    public static boolean isOpen(int i, int j) {
        check_bounds(i);
        check_bounds(j);
        if (i == 1) {
            // StdOut.println("Checking " + 0 + "<->" + j);
            return uf.connected(0, j);
        }
        else {
            // StdOut.println("Checking " + ((i-2)*N + j) + "<->" + ((i-1)*N + j));
            return uf.connected((i-2)*N + j, (i-1)*N + j);
        }
    }

    // is site (row i, column j) full or blocked or disconnected?
    public static boolean isFull(int i, int j) {
        return !isOpen(i, j);
    }

    // does the system percolate?
    public boolean percolates() {
        return uf.connected(0, N*N+1);
    }

    // punch a random hole (site) in the matrix
    public static boolean random_hole() {
        int row = (int) floor(N*random() + 1.0);
        int col = (int) floor(N*random() + 1.0);
        if (isFull(row, col)) {
            open(row, col);
            // StdOut.println("Opened " + row + ", " + col);
            return true;
        }
        else
            return false;
    }

    // test client, optional
    public static void check_one_case(String[] args) {
        N = 8;
        int attempts = 0;
        if (args.length > 0) {
            N = Integer.parseInt(args[0]);
        }
        Percolation p = new Percolation(N);
        for (int i=1; i<args.length; i++) {
            int num_holes = Integer.parseInt(args[i]);
            for (int j=0; j<num_holes; j++) {
                attempts = 0;
                while ((!random_hole()) && (attempts < max_attempts) && (j < num_holes)) {
                    attempts += 1;
                    // StdOut.println("attempt " + j + "." + attempts);
                    }
                // StdOut.println(j);
            }
            StdOut.println("With " + num_holes + " holes in a " + N + "x" + N + " matrix, percolates ?= " + p.percolates());
        }
    }


    // test client, optional
    public static void main(String[] args) {
        N = 8;
        int attempts = 0;
        if (args.length > 0) {
            N = Integer.parseInt(args[0]);
        }
        Percolation p = new Percolation(N);
        int num_holes = 0;
        while ((num_holes<N*N) && (!p.percolates())) {
            num_holes += 1;
            while ((!random_hole()) && (attempts < max_attempts)) {
                attempts += 1;
                // StdOut.println("attempt " + j + "." + attempts);
            }
            // StdOut.println(num_holes);
        }
        StdOut.println("With " + num_holes + " holes in a " + N + "x" + N + " matrix ("+ (100.0 * num_holes / N / N) + "%), percolates ?= " + p.percolates());
    }
}