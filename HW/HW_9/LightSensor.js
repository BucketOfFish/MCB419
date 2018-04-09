class LightSensor {
  constructor(name, x, y) {
    this.name = name;
    this.x = x;
    this.y = y;
    this.output = 0;
    this.outgoing = [];
  }


  display() {
    push();
    stroke(250);
    fill(250);
    translate(this.x, this.y);
    line(0, 0, 0, -6);
    line(0, -6, 6, -12);
    line(0, -6, -6, -12);
    if (showLabels) {
      textAlign(CENTER, CENTER);
      textSize(10);
      noStroke();
      text(this.name, 0, 10);
    }
    pop();
  }

  update() {
    this.output = 0;
    // in this model, the sensor only responds to light above the sensor
    if (light.y < this.y) {
      var d = max(20, dist(this.x, this.y, light.x, light.y));
      this.output = light.intensity / (d * d);
    }
  }
}