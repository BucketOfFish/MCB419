class World {
  constructor() {
    this.lights = [];
    this.lights.push(new Light(width / 2, height / 2, 100));
    this.scoring = new Scoring();
  }

  display() {
    background(0);
    for (let light of this.lights) light.display();
    this.scoring.display();
  }
}

// Scoring is used to score bot performance

class Scoring {
  constructor() {
    this.reset();
  }

  reset() {
    this.total = 0;
    this.passed = false;
    this.failed = false;
    for (let b of bots) {
      b.pts = 0;
      b.redEligible = false;
      b.blueEligible = false;
      b.yellowEligible = true;
    }
    // radii for scoring circles
    this.radii = {
      red: 40,
      blue: 80,
      yellow: 160
    };
  }
  
  display() {
    // draw target rings
    push();
    translate(width / 2, height / 2);
    stroke(255);
    noFill();
    for (let cname of ['red', 'blue', 'yellow']) {
      stroke(cname);
      ellipse(0, 0, 2 * this.radii[cname]);
    }

    textAlign('center');
    textSize(60);
    noStroke();
    if (this.passed) {  
      fill('green');
      text('PASSED', 0, 0);
    } else if (this.failed) {
      fill('red');
      text('FAILED', 0, 0);
    }
    
    pop();
  }

  update() {
    let mode = select("#controller").value();
    switch (mode) {

      case 'aggressive':
        // earn points for hitting the light
        for (let b of bots) {
          let rbot = dist(b.x, b.y, width / 2, height / 2);
          // bot must leave yellow before becoming eligible to score
          if (rbot > this.radii.yellow) b.redEligible = true;
          // check if bot is inside red circle
          if (b.redEligible && rbot < this.radii.red) {
            // bot is inside red circle and eligible to score
            b.pts += 1;
            b.redEligible = false;
          } 
        }
        break;

      case 'coward':
        // earn points for staying outside of yellow circle and moving slow
        for (let b of bots) {
          let rbot = dist(b.x, b.y, width / 2, height / 2);
          if (rbot > this.radii.yellow) {
            if ((b.mtrL + b.mtrR) < 0.5) b.pts += 0.005;
          } else {
            // reset pts if bot enters yellow circle
            b.pts = 0;
          }
        }

        break;

      case 'explorer':
        // earn points for moving quickly outside yellow 
        // entering blue and not entering red 
        for (let b of bots) {
          let rbot = dist(b.x, b.y, width / 2, height / 2);
          if (rbot > this.radii.yellow) {
            b.blueEligible = true;
            // pts for moving fast outside of yellow (only scored once)
            if(b.yellowEligible && ((b.mtrL + b.mtrR) > 4)) {
              b.pts += 1;
              b.yellowEligible = false;
            }
          }
          if (b.blueEligible && rbot < this.radii.blue) {
            // pts for entering blue, not eligible again until leaves yellow
            b.pts += 1;
            b.blueEligible = false;
          }
          if (rbot < this.radii.red) {
            // red violation, this bot must start over
            b.pts = 0.0;
            b.yellowEligible = true;
            b.blueEligible = false;
          }
        }
        break;

      case 'love':
        // earn points for stopping between red and blue circles
        // 2 points per bot
        for (let b of bots) {
          let rbot = dist(b.x, b.y, width / 2, height / 2);
          if (rbot > this.radii.red && rbot < this.radii.blue &&
            abs(b.mtrL + b.mtrR) < 0.01) {
            b.pts = 2;
          } else {
            b.pts = 0;
          }
        }
        break;
    }

    // sum individual bot points to get total score
    this.total = 0;
    for (let b of bots) this.total += b.pts;

    // criterion for "passing" 
    if (this.total >= 8) this.passed = true;
    
    // criteria for "failing"
    if (itick >= 1000 && !this.passed)this.failed = true;
  }
}