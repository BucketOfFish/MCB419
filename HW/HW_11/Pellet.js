class Pellet {
  constructor(pColor) {
    this.x = random(10, width - 10);
    this.y = random(10, height - 10);
    this.intensity = 1000;
    this.color = pColor;
    this.value = this.getNewValue();
  }

  display() {
    if (this.intensity === 0) {
      // invisible
      stroke(0);
      noFill();
    } else {
      // visible
      noStroke();
      fill(this.color);
    }
    ellipse(this.x, this.y, 8, 8);
  }

  getNewValue() {
    let valueGood = 4;
    let valueBad = -4;
    let ival = -1;
    if (this.color == valueOrder[0]) ival = 0;
    if (this.color == valueOrder[1]) ival = 1;
    if (this.color == valueOrder[2]) ival = 2;

    let probBad = 0.1 + 0.4 * ival; // [.1 .5 .9]
    let value = (random(1) < probBad) ? valueBad : valueGood;
    return value;
  }
}