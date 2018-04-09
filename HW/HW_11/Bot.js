//
//  Braitenberg-style vehicle with FSM support
//

class Bot {

  //===========================================
  // MODIFY updateEstimates() and seekUser()
  //===========================================

  updateEstimates() {
    // this gets called on every tick
    // update the estimatedValue array when the bot consumes a pellet
    // use the delta rule; pick a good learning rate
    let red = 0;
    let green = 1;
    let blue = 2;
    let learning_rate = 0.1;
    let consumed = -1;
    if (bot.sns.lastColorConsumed == "red") consumed = 0;
    if (bot.sns.lastColorConsumed == "green") consumed = 1;
    if (bot.sns.lastColorConsumed == "blue") consumed = 2;
    let dValue = (bot.sns.deltaEnergy - this.estimatedValue[consumed]);
    if (bot.sns.deltaEnergy != 0.0) this.estimatedValue[consumed] = this.estimatedValue[consumed] + learning_rate * dValue;
  }

  seekUser() {
    // your controller code goes here
    // use the estimatedValue array to decide which pellets to consume
    // you can call other predefined controller functions, if desired
    // (e.g., you can call "this.seekRed()", "this.seekGreen()", etc.
    let seek = random();
    let beta = 3;
    let redZ = exp(beta * this.estimatedValue[0]);
    let greenZ = exp(beta * this.estimatedValue[1]);
    let blueZ = exp(beta * this.estimatedValue[2]);
    let Z = redZ + greenZ + blueZ;
    let redRatio = redZ / Z;
    let greenRatio = greenZ / Z;
    let blueRatio = blueZ / Z;
    if (seek < redRatio) this.seekRed();
    else if (seek < redRatio + greenRatio) this.seekGreen();
    else this.seekBlue();
  }


  //=========================================
  // DO NOT CHANGE THE FOLLOWING CONTROLLERS
  //=========================================


  seekAll() {
    var snsL = this.sns.left[0] + this.sns.left[1] + this.sns.left[2];
    var snsR = this.sns.right[0] + this.sns.right[1] + this.sns.right[2];
    if (snsL > snsR) {
      this.mtr.left = 0;
      this.mtr.right = 10;
    } else {
      this.mtr.right = 0;
      this.mtr.left = 10;
    }
  }

  seekRed() {
    if (this.sns.left[0] > this.sns.right[0]) {
      this.mtr.left = 0;
      this.mtr.right = 10;
    } else {
      this.mtr.right = 0;
      this.mtr.left = 10;
    }
  }

  seekGreen() {
    if (this.sns.left[1] > this.sns.right[1]) {
      this.mtr.left = 0;
      this.mtr.right = 10;
    } else {
      this.mtr.right = 0;
      this.mtr.left = 10;
    }
  }

  seekBlue() {
    if (this.sns.left[2] > this.sns.right[2]) {
      this.mtr.left = 0;
      this.mtr.right = 10;
    } else {
      this.mtr.right = 0;
      this.mtr.left = 10;
    }
  }

  //======================
  // END OF CONTROLLERS
  //======================

  constructor() {
    this.dia = 25;
    this.cfill = 'darkOrange';
    this.reset();
  }

  consume() {
    this.sns.deltaEnergy = 0.0;
    let bx = this.x; // bot x
    let by = this.y; // bot y
    let rsq = bot.dia * bot.dia / 4;
    for (let i = pellets.length - 1; i >= 0; i--) {
      if ((pellets[i].x - bx) * (pellets[i].x - bx) + (pellets[i].y - by) * (pellets[i].y - by) < rsq) {
        this.energy += pellets[i].value; // bot "eats" the pellet, gains energy
        this.sns.deltaEnergy += pellets[i].value; // sense change in energy
        this.sns.lastColorConsumed = pellets[i].color;
        pellets[i].x = random(10, width-10);     // reposition the pellet
			  pellets[i].y = random(10, height-10);
			  pellets[i].value = pellets[i].getNewValue();
      }
    }
  }

  display() {
    // Braitenberg bugs
    push();
    translate(this.x, this.y);
    // text labels
    noStroke();
    fill(0);
    let xtxt = 20;
    let ytxt = 20;
    textAlign(LEFT);
    if (this.x > 0.8 * width) {
      xtxt = -20;
      textAlign(RIGHT);
    }
    if (this.y > 0.8 * height) {
      ytxt = -20;
    }

    let controllerString = this.controllerName;
    if (this.controllerName.substring(0, 3) === "fsm") { // FSM
      controllerString += ": " + this.state;
    }
    text(controllerString, xtxt, ytxt);
    var energyString = "energy: " + nf(this.energy, 0, 0);
    text(energyString, xtxt, ytxt + 15);

    // draw body
    stroke(0);
    fill(this.cfill);
    ellipse(0, 0, this.dia);
    // draw head
    rotate(this.heading);
    ellipse(12, 0, 8, 12);
    // eyes
    fill(0);
    ellipse(14, 5, 4);
    ellipse(14, -5, 4);

    pop();
  }


  reset() {
    this.x = width / 2;
    this.y = height / 2;
    this.heading = 0;
    this.energy = 0;
    this.state = ''; // FSM
    this.counter = 0; // FSM
    this.estimatedValue = [0, 0, 0]; // estimated value of R, G, B pellets

    this.sns = {
      left: [0, 0, 0], // activation level for left sensor [R, G, B]
      right: [0, 0, 0], // activation level for right sensor [R, G, B]
      collision: false, // true = hit wall/obstacle, 
      deltaEnergy: 0, // energy gained on last time step
      lastColorConsumed: "black", // color of the last pellet that was consumed ("red", "green", "blue")
    };

    this.mtr = {
      left: 0, // activation level for left motor
      right: 0 // activation level for right motor
    };

    let cname = select("#controller").value();
    this.setController(cname);
  }


  update() {

    this.updateSensors();

    // constrain after updating sensors to allow collision detection
    let r = this.dia / 2;
    this.x = constrain(this.x, r, width - r);
    this.y = constrain(this.y, r, height - r);

    this.updateEstimates();
    this.consume();
    this.controller();

    let newSpeed = constrain((this.mtr.left + this.mtr.right) / 2.0, -5.0, 5.0);
    this.heading += constrain((this.mtr.left - this.mtr.right) / this.dia, -0.2, 0.2);
    this.heading %= TWO_PI;
    this.x += newSpeed * cos(this.heading);
    this.y += newSpeed * sin(this.heading);

  }

  updateSensors() {

    let bx = this.x;
    let by = this.y;
    let r = this.dia / 2;
    this.sns.collision = (bx < r) || (bx > width - r) || (by < r) || (by > height - r);

    this.sns.left = [0, 0, 0];
    this.sns.right = [0, 0, 0];

    if (pellets === undefined || pellets.length === 0) {
      return;
    }

    // compute sensor locations
    let dr = 0.7 * this.dia; // sensor from center of bot
    let dtheta = PI / 4; // angle relative to midline
    let xL = bx + dr * Math.cos(this.heading - dtheta);
    let yL = by + dr * Math.sin(this.heading - dtheta);
    let xR = bx + dr * Math.cos(this.heading + dtheta);
    let yR = by + dr * Math.sin(this.heading + dtheta);

    let ux = Math.cos(this.heading);
    let uy = Math.sin(this.heading);

    // compute intensity at sensor location
    for (let i = 0; i < pellets.length; i++) {
      var px = pellets[i].x;
      var py = pellets[i].y;
      var pc = pellets[i].color;
      var colorIdx = 0; // red
      if (pc == "green") {
        colorIdx = 1;
      }
      if (pc == "blue") {
        colorIdx = 2;
      }
      var pinten = pellets[i].intensity;
      // only sense pellets that are in front of the bot
      var dot = (px - this.x) * ux + (py - this.y) * uy;
      if (dot > 0) {
        var distLsq = (px - xL) * (px - xL) + (py - yL) * (py - yL);
        if (distLsq < 1) {
          distLsq = 1;
        }
        var distRsq = (px - xR) * (px - xR) + (py - yR) * (py - yR);
        if (distRsq < 1) {
          distRsq = 1;
        }
        var sl = pinten / distLsq;
        var sr = pinten / distRsq;
        this.sns.left[colorIdx] += sl;
        this.sns.right[colorIdx] += sr;
      }
    }
  }

  setController(name) {
    this.controllerName = name;
    this.controller = this[name];
    if (name.substring(0, 3) === 'fsm') { // FSM
      this.controller('init');
    }
  }

  transitionTo(state) { // FSM
    this.counter = 0;
    this.state = state;
  }

}