public class PercolationTest {
    private static int N = 8;
    private static Percolation perc;
    private static double maxN = Math.floor(Math.sqrt(1.0e9));
    private static double max_attempts = maxN;
    private static StdRandom random;

    // punch a random hole (permiable site) in the matrix
    private static boolean randomHole(Percolation perc) {
        int row = StdRandom.uniform(N) + 1;
        int col = StdRandom.uniform(N) + 1;
        if (perc.isOpen(row, col))
            return false;
        else {
            perc.open(row, col);
            // StdOut.println("Opened " + row + ", " + col);
            return true;
        }
    }

    // random trial to see how many holes required to percolate:
    private static int holesToPercolate(int i, Percolation perc) {
        int numHoles = 0;
        while ((numHoles < i * i) && (!perc.percolates())) {
            numHoles += 1;
            int attempts = 0;
            while (!randomHole(perc)) {
                attempts += 1;
                // StdOut.println("attempt " + j + "." + attempts);
            }
            // StdOut.println(numHoles);
        }
        if (perc.percolates())
            return numHoles;
        return (-1 * numHoles);
    }

    public static void print_perc(Percolation perc) {
        for (int i=0; i<N; i++) {
            for (int j=0;j<N;j++) {
                if (perc.isOpen(i+1, j+1)) {
                    if (perc.isFull(i+1, j+1))
                        StdOut.print("*");
                    else
                        StdOut.print(".");
                }
                else
                    StdOut.print("O");
            }
            StdOut.println("");
        }
    }

    // test client, optional
    public static void main(String[] args) {
        N = 8;
        if (args.length > 0) {
            N = Integer.parseInt(args[0]);
        }
        perc = new Percolation(N);
        StdOut.println("isFull(1,1) = " + perc.isFull(1,1));
        StdOut.println("isFull(2,1) = " + perc.isFull(2,1));
        StdOut.println("isFull(2,N) = " + perc.isFull(2,N));
        int num_holes = holesToPercolate(N, perc);

        if (num_holes > 0) 
            StdOut.println("Percolates with " + num_holes + " holes in a " + N + "x" + N + " matrix ("+ (100.0 * num_holes / N / N) + "%).");
        else
            StdOut.println("Unable to percolate with " + (-num_holes) + " holes in a " + N + "x" + N + " matrix ("+ (-100.0 * num_holes / N / N) + "%).");
    }
}