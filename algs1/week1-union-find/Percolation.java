public class Percolation extends CompressedQuickUnionUF {
    public int sqrtN = 8;

    // create N-by-N grid, with all sites blocked
    public Percolation(int i) {
        sqrtN = i;
        // node 0 is the "top" or input node that is connected to the input surface
        // node N+1 is the "bottom" or output node that is connected to the output surface
        // node 1 is the upper left node 
        // node N is the upper right node
        // node N+1 is the left-hand node of the second row
        super(sqrtN*sqrtN+2);
    }

    public void check_bounds(int i) {
        if (i < 0 || i > sqrtN) {
            throw new IndexOutOfBoundsException("Index " + i " is outside the range 1 to " + sqrtN);
        }
    }
    // open site (row i, column j) if it is not already
    public void open(int i, int j) {
        check_bounds(i);
        check_bounds(j);

    }
    // is site (row i, column j) open?
    public boolean isOpen(int i, int j) {
        check_bounds(i);
        check_bounds(j);

    }
    // is site (row i, column j) full?
    public boolean isFull(int i, int j) {
        check_bounds(i);
        check_bounds(j);

    } 
    // does the system percolate?
    public boolean percolates() {

    }

    // test client, optional
    // public static void main(String[] args) {}
}