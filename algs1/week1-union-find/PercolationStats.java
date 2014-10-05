public class PercolationStats {
    private double[] thresholds;
    private double nn;
    private int N;


    // perform T independent computational experiments on an N-by-N grid
    public PercolationStats(int N, int T) {
        this.N = N;
        if ((N <= 0) || (T <= 0))
            throw new IllegalArgumentException("Either N (" + N + ") or T (" + T + ") is invalid (<=0)");
        nn = N * N;
        thresholds = new double[T];
        Percolation perc = new Percolation(N);
        for (int i = 0; i < T; i++) {
            thresholds[i] = holesToPercolate(N, perc) / nn;
        }
    }

    // punch a random hole (permiable site) in the matrix
    private boolean randomHole(Percolation perc) {
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
    private int holesToPercolate(int i, Percolation perc) {
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

    // sample mean of percolation threshold
    public double mean() {
        return StdStats.mean(thresholds);
        // double sum = 0.0;
        // for (double thresh : thresholds)
        //     sum += thresh;
        // return sum / (double) thresholds.length;
    }

    // returns lower bound of the 95% confidence interval  
    public double confidenceLo() {
        double mu = mean();
        double std = stddev();
        return mu - 1.96 * std / Math.sqrt((double) thresholds.length);
    }

    // sample standard deviation of percolation threshold
    public double stddev() {
        // double mu = mean();
        return StdStats.stddev(thresholds);
        // double sum = 0.0;
        // for (double thresh : thresholds)
        //     sum += (thresh - mu) * (thresh - mu);
        // return sum / (double) (thresholds.length - 1);
    }

    // returns upper bound of the 95% confidence interval
    public double confidenceHi() {
        double mu = mean();
        double std = stddev();
        return mu + 1.96 * std / Math.sqrt((double) thresholds.length);
    }

    // test client, described belowf
    public static void main(String[] args) {
        PercolationStats ps = new PercolationStats(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
        // ps.N = N;
        // ps.T = T;
        double mu = ps.mean();
        double std = ps.stddev();
        String msg = "";
        msg += "mean                    = " + mu + "\n";
        msg += "stddev                  = " + std + "\n";
        msg += "95% confidence interval = " + ps.confidenceLo() + ", " + ps.confidenceHi() + "\n";
        StdOut.println(msg);
    }
}
