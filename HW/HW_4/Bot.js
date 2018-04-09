class Bot {

  constructor() {
    // THIS CODE IS OK... NO CHANGES
    this.reset();
  }

  reset() {
    // THIS CODE IS OK... NO CHANGES
    this.x = random(width);
    this.y = random(height);
    this.r = 6;
    this.heading = random(TWO_PI);
    this.speed = 1;
    this.turnAngle = 0;
    this.controller = this.wander;
    this.memory = world.getTemperature(this.x, this.y);  // NEW: use this variable to remember something useful
  }

  update() {
    // THIS CODE IS OK... NO CHANGES

    // change heading based on value of wanderNoise
    this.heading += this.turnAngle;

    // update position
    this.x += this.speed * cos(this.heading);
    this.y += this.speed * sin(this.heading);

    // wrapped boundary conditions
    this.x = (this.x + width) % width;
    this.y = (this.y + height) % height
  }

  display() {
    // THIS CODE IS OK... NO CHANGES
    push()
    translate(this.x, this.y);
    rotate(this.heading);
    noStroke();
    fill(250, 200, 150);
    ellipse(0, 0, 2 * this.r, this.r);
    stroke(0);
    line(0, 0, this.r, 0);
    pop();
  }

  //==============================================================
  // CONTROLLER CODE STARTS HERE - YOU NEED TO EDIT THIS CODE
  // each controller should set this.speed and this.turnAngle
  //==============================================================

  wander() {
    this.speed = 1;
    this.turnAngle = 0.1 * random(-1, 1);
  }
  
  orthokinesisPos() {
    // speed should increase with temperature
    let tsns = world.getTemperature(this.x, this.y);
    var threshold = 0.15;
    if (tsns > threshold) this.speed = 1;
    else this.speed = 0;
    this.turnAngle = 0.1 * random(-1, 1); // do not change this line
  }
  
  orthokinesisNeg() {
    // speed should decrease with temperature
    let tsns = world.getTemperature(this.x, this.y);
    var threshold = 0.85;
    if (tsns < threshold) this.speed = 1;
    else this.speed = 0;
    this.turnAngle = 0.1 * random(-1, 1); // do not change this line
  }
  
  runTumble() {
    // run (go straight) if things are getting better, otherwise tumble
    var tsns = world.getTemperature(this.x, this.y);
    this.speed = 1; // do not change this line
    if (tsns > this.memory) {
	    this.turnAngle = 0;
    }
    else {
      this.turnAngle = PI * random(-1, 1);
    }
    this.memory = tsns;
  }
  
  klinokinesisPos() {
    // turnAngle should increase with temperature (can be nonlinear)
    let tsns = world.getTemperature(this.x, this.y);
    this.speed = 1; // do not change this line;
    if (tsns > 0.8)
    	this.turnAngle = PI * random(-1, 1);
  }
  
  klinokinesisNeg() {
    // turnAngle should decrease with temperature (can be nonlinear)
    let tsns = world.getTemperature(this.x, this.y);
    this.speed = 1 // do not change this line
    if (tsns < 0.2)
    	this.turnAngle = PI * random(-1, 1);
  }
  
}