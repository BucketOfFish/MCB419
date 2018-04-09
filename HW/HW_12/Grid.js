class Grid {
  constructor(nx, ny, cellSize) {
    this.nx = nx;
    this.ny = ny;
    this.cellSize = cellSize; // pixels
    this.dat = new Array(nx);
    for (let i = 0; i < nx; i++) {
      this.dat[i] = new Array(ny);
      for (let j = 0; j < ny; j++) {
        this.dat[i][j] = 0;
      }
    }
  }
  
  display() {
    var cs = this.cellSize;
    for (let i = 0; i < this.nx; i++) {
      for (let j = 0; j < this.ny; j++) {
        let dat = this.dat[i][j];
        if (dat == 1) { // wall
          fill(128);
          rect(i * cs, j * cs, cs, cs);
        } else if (dat == 2) { // good pellet (green)
          fill('green');
          ellipse((i + 0.5) * cs, (j + 0.5) * cs, 0.5 * cs, 0.5 * cs);
        } else if (dat == 3) { // bad pellet (blue)
          fill('blue');
          ellipse((i + 0.5) * cs, (j + 0.5) * cs, 0.5 * cs, 0.5 * cs);
        }
      }
    }
  }
  
  getEmptyCell() {
    // return {ix, iy} of an empty cell
    while (true) {
      var ix = randint(0, this.nx - 1);
      var iy = randint(0, this.ny - 1);
      if (this.get(ix, iy) === 0) return {ix, iy};
    }
  }

  get(ix, iy) {
    return this.dat[ix][iy];
  }
  
  set(ix, iy, ival) {
    this.dat[ix][iy] = ival;
  } 
}