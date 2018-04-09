class Motor {
  constructor(name, x, y) {
    this.name = name;
    this.x = x;
    this.y = y;
    this.angle = 0;
    this.angular_vel = 0;
    this.incoming = [];
  }

  display() {
    // draw the wheel
    push();
    stroke(250, 200, 0);
    fill(50);
    translate(this.x, this.y);
    ellipse(0, 0, 30, 30);
    if (showLabels) {
      textAlign(CENTER, CENTER);
      textSize(10);
      fill(250);
      noStroke();
      text(this.name, 0, 25);
      text("vel: " + nf(this.angular_vel, 0, 2), 0, 40);
    }
    // rotation indicators
    rotate(this.angle);
    stroke(125, 100, 0);
    line(-14, 0, 14, 0);
    line(0, -14, 0, 14);
    pop();
  }

  update() {
    // each incoming spike increments the angular velocity in proportion to the synaptic weight
    for (let i = 0; i < this.incoming.length; i++) {
      var axon = this.incoming[i];
      this.angular_vel += axon.weight * axon.pre.output;
    }
    this.angular_vel *= 0.98; // velocity decay
    this.angle += this.angular_vel; // angle update
    this.angle %= TWO_PI; // wrap angle
  }
}