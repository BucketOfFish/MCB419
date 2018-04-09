class Bot {
  constructor() {
    this.reset();
    this.genes = [0, 0, 0, 0, 0, 0];
  }

  consume() {
    let rsq = (this.dia / 2) * (this.dia / 2);
    for (let p of pellets) {
      let dsq = (p.x - this.x) * (p.x - this.x) + (p.y - this.y) * (p.y - this.y);
      if (dsq < rsq) {
        // consume pellet
        this.fitness++;
        p.reset();
      }
    }
  }

  reset() {
    this.x = width / 2;
    this.y = height / 2;
    this.heading = 0;
    this.dia = 20;
    this.snsL = 0;
    this.snsR = 0;
    this.mtrL = 0;
    this.mtrR = 0;
    this.fitness = 0;
    this.updateSensors();
  }

  update() {
    // controller has already set values of mtrL and mtrR
    this.mtrL = this.genes[0] + this.genes[1] * this.snsL + this.genes[2] * this.snsR;
    this.mtrR = this.genes[3] + this.genes[4] * this.snsL + this.genes[5] * this.snsR;
    
    let speed = constrain((this.mtrL + this.mtrR) / 2.0, -5.0, 5.0);
    let turnAngle = constrain((this.mtrL - this.mtrR) / this.dia, -0.4, 0.4);

    this.heading += turnAngle;
    this.heading %= TWO_PI;

    this.x += speed * cos(this.heading);
    this.y += speed * sin(this.heading);

    // solid boundary conditions
    let dd = 0.7 * this.dia;
    this.x = constrain(this.x, dd, width - dd);
    this.y = constrain(this.y, dd, height - dd);
  }

  updateSensors() {
    // pellets is a global array with an (x,y) and intensity value
    // sensor activation is based on inverse-square law
    this.snsL = 0;
    this.snsR = 0;

    // compute sensor locations
    let dr = 0.7 * this.dia; // sensor from center of bot
    let dtheta = PI / 4; // angle relative to midline
    let xL = this.x + dr * Math.cos(this.heading - dtheta);
    let yL = this.y + dr * Math.sin(this.heading - dtheta);
    let xR = this.x + dr * Math.cos(this.heading + dtheta);
    let yR = this.y + dr * Math.sin(this.heading + dtheta);

    let ux = Math.cos(this.heading);
    let uy = Math.sin(this.heading);
    
    for (let p of pellets) {
      let distp = Math.sqrt((p.x - this.x) * (p.x - this.x) + (p.y - this.y) * (p.y - this.y));
      let dotprod = (p.x - this.x) * ux + (p.y - this.y) * uy;
      let cosang = dotprod / distp;

      if (cosang > 0.5) {
        let distLsq = (p.x-xL)*(p.x-xL) + (p.y-yL)*(p.y-yL);
        if (distLsq < 1) distLsq = 1;
        let distRsq = (p.x-xR)*(p.x-xR) + (p.y-yR)*(p.y-yR);
        if (distRsq < 1) distRsq = 1;
        
        this.snsL += p.intensity / distLsq;
        this.snsR += p.intensity / distRsq;
      }
    }
  }

  display() {
    // Braitenberg bugs
    push();
    translate(this.x, this.y);
    rotate(this.heading);

    //draw "wheels"
    stroke(100);
    strokeWeight(0.3 * this.dia);
    line(-0.2 * this.dia, 0.8 * this.dia, 0.1 * this.dia, 0.8 * this.dia);
    line(-0.2 * this.dia, -0.8 * this.dia, 0.1 * this.dia, -0.8 * this.dia);

    // draw axle
    strokeWeight(1);
    line(-0.1 * this.dia, 0.8 * this.dia, -0.1 * this.dia, -0.8 * this.dia);

    // draw "body"
    fill("brown");
    ellipse(0, 0, this.dia, this.dia);

    // draw "sensors"
    fill(100);
    var d = 0.3 * this.dia;
    ellipse(this.dia / 2, this.dia / 2, d);
    ellipse(this.dia / 2, -this.dia / 2, d);
    pop();
  }

}