class Trail {

  constructor(maxPoints, trailColor) {
    this.maxPoints = maxPoints;
    this.trailColor = trailColor;
    this.reset();
  }
  
  reset() {
    this.index = 0;
    this.full = false;
    this.xtrail = new Array(this.maxPoints);
    this.ytrail = new Array(this.maxPoints);
  }

  update(x, y) {
    this.xtrail[this.index] = x;
    this.ytrail[this.index] = y;
    this.index++;
    if (this.index == this.maxPoints) {
      this.index = 0;
      this.full = true;
    }
  }

  display() {
    stroke(this.trailColor);
    let npts = this.full ? this.maxPoints : this.index;
    for (var i = 0; i < npts; i++) {
      point(this.xtrail[i], this.ytrail[i]);
    }
  }
}