class Bot {

  constructor() {
    // CHANGE THESE VALUES TO MATCH SPECS
    this.x = width/2;
    this.y = height/2;
    this.speed = 1;
    this.heading = 0;
    this.r = 15;
    this.wanderNoise = 0.0;
  }

  consume() {
    // ADD CODE HERE TO CONSUME PELLETS
    for (var i=pellets.length-1; i>=0; i--) {
      var p = pellets[i];
    	var distance = sqrt(pow(p.x - this.x, 2) + pow(p.y - this.y, 2));
      if (distance <= this.r) {
      	pellets.splice(i, 1);
      }
    }
  }

  reset() {
    // CHANGE THIS
    this.x = width/2;
    this.y = height/2;
  }

  update() {
    // THIS CODE IS OK... NO CHANGES
    
    // change heading based on value of wanderNoise
    this.heading += this.wanderNoise * random(-1, 1);

    // update position
    this.x += this.speed * cos(this.heading);
    this.y += this.speed * sin(this.heading);

    // boundary conditions
    if (wrapped) {
      this.x = (this.x + width) % width;
      this.y = (this.y + height) % height
    } else {
      // solid walls
      this.x = constrain(this.x, this.r, width - this.r);
      this.y = constrain(this.y, this.r, height - this.r);
    }
  }

  display() {
    // THIS CODE IS OK... NO CHANGES
    push()
    translate(this.x, this.y);
    rotate(this.heading);
    noStroke();
    fill(250, 200, 150);
    ellipse(0, 0, 2 * this.r);
    stroke(0);
    line(0, 0, this.r, 0);
    pop();
  }
}