class Pellet {
  constructor() {
    this.reset();
  }
  
  reset() {
    this.x = random(10, width - 10);
    this.y = random(10, height - 10);
    this.intensity = 1000;
  }
  
  display() {
    noStroke();
    fill('green');
    ellipse(this.x, this.y, 6);
  }
}