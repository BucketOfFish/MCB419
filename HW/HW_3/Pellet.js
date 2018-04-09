class Pellet {
  constructor() {
    // ADD CODE HERE
    this.x = random(width);
    this.y = random(height);
  }
  
  display() {
    // ADD CODE HERE
    stroke('lightGreen');
    point(this.x, this.y);
  }
}