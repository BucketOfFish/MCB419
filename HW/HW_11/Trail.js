class Trail {
  constructor(maxPoints = 100, defaultColor = 'grey') {
    this.maxPoints = maxPoints;
    this.defaultColor = defaultColor;
    this.index = 0;
    this.full = false;
    this.xtrail = new Array(maxPoints);
    this.ytrail = new Array(maxPoints);
    this.color = new Array(maxPoints);
  }
  
  reset() {
    this.index = 0;
    this.full = false;
  }

  update(x, y, trailColor) {
    this.xtrail[this.index] = x;
    this.ytrail[this.index] = y;
    if(trailColor !== undefined){
      this.color[this.index] = trailColor;
    } else {
      this.color[this.index] = this.defaultColor;
    }
    this.index++;
    if (this.index == this.maxPoints) {
      this.index = 0;
      this.full = true;
    }
  }

  display() {
    push();
    let npts = this.full ? this.maxPoints : this.index;
    for (let i = 0; i < npts; i++) {
      fill(this.color[i]);
      ellipse(this.xtrail[i], this.ytrail[i], 3);
    }
    pop();
  }
}