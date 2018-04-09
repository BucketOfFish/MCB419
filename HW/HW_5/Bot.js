class Bot {

  constructor(x=random(width), y=random(height)) {
    this.x = x;
    this.x0 = this.x;
    this.y = y;
    this.y0 = this.y;
    this.major = 25;
    this.minor = 15;
    this.cfill = 'grey';
    this.trail = new Trail(150, this.cfill);
    this.reset();
  }

  reset() {
    this.x = this.x0;
    this.y = this.y0;
    this.heading = random(TWO_PI);
    this.snsL = 0;
    this.snsR = 0;
    this.mtrL = 0;
    this.mtrR = 0;
    this.trail.reset();
  }

  update() {
    // controller has already set values of mtrL and mtrR

    let newSpeed = constrain((this.mtrL + this.mtrR) / 2.0, -5.0, 5.0);
    let turnAngle = constrain((this.mtrL - this.mtrR) / this.minor, -0.1, 0.1);

    this.heading += turnAngle;
    this.heading %= TWO_PI;

    this.x += newSpeed * cos(this.heading);
    this.y += newSpeed * sin(this.heading);

    // wrapped boundary conditions
    this.x = (this.x + width) % width;
    this.y = (this.y + height) % height

    this.trail.update(this.x, this.y);
  }

  updateSensors() {
    // compute sensor values
    this.snsL = 0;
    this.snsR = 0;
    let dr = mag(this.major, this.minor); // distance from center of bot
    let dtheta = atan2(this.minor, this.major); // angle relative to midline
    let xL = this.x + dr * cos(this.heading - dtheta);
    let yL = this.y + dr * sin(this.heading - dtheta);
    let xR = this.x + dr * cos(this.heading + dtheta);
    let yR = this.y + dr * sin(this.heading + dtheta);
    for (let i = 0; i < world.lights.length; i++) {
      let light = world.lights[i];
      let distL = max(1, dist(light.x, light.y, xL, yL));
      let distR = max(1, dist(light.x, light.y, xR, yR));
      let inten = max(0, light.intensity); // ignore negative values
      this.snsL += inten / sq(distL);
      this.snsR += inten / sq(distR);
    }
  }

  display() {
    // THIS CODE IS OK... NO CHANGES
    // Braitenberg bugs

    this.trail.display();

    push();
    translate(this.x, this.y);
    rotate(this.heading);

    //draw "wheels"
    stroke(150, 200);
    strokeWeight(0.3 * this.minor);
    line(-0.5 * this.major, 0.8 * this.minor, -0.1 * this.major, 0.8 * this.minor);
    line(-0.5 * this.major, -0.8 * this.minor, -0.1 * this.major, -0.8 * this.minor);

    // draw axle
    strokeWeight(1);
    line(-0.3 * this.major, 0.8 * this.minor, -0.3 * this.major, -0.8 * this.minor);

    // draw "body"
    fill(this.cfill);
    ellipse(0, 0, this.major, this.minor);

    // draw "sensors"
    fill(150);
    ellipse(this.major / 2, this.minor / 2, this.minor / 3);
    ellipse(this.major / 2, -this.minor / 2, this.minor / 3);
    pop();

  }

  //==============================================================
  // CONTROLLER CODE STARTS HERE - YOU NEED TO EDIT THIS CODE
  // updateSensors has already been called, you don't need to call it here
  //==============================================================

  aggressive() {
    this.mtrL = 5 + 15 * this.snsR;
    this.mtrR = 5 + 15 * this.snsL;
  }

  coward() {
    this.mtrL = 15 * this.snsL;
    this.mtrR = 15 * this.snsR;
  }

  explorer() {
    this.mtrL = 5 - 5 * this.snsR;
    this.mtrR = 5 - 5 * this.snsL;
  }

  love() {
    this.mtrL = 3 - 15 * this.snsL;
    this.mtrR = 3 - 15 * this.snsR;
  }
}