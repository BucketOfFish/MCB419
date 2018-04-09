//
//  Braitenberg-style vehicle with FSM support
//

class Bot {

  //================================
  // MODIFY handCoded() CONTROLLER
  //================================

  handCoded() {

    // sample code - you need to modify this routine

    var snsFWD = this.sensors[FWD].val;
    var snsLEFT = this.sensors[LEFT].val;
    var snsRIGHT = this.sensors[RIGHT].val;
    
    if (snsFWD == obj.good) {
      this.action = FWD;
    } 
    else if (snsLEFT == obj.good) {
      this.action = LEFT;
    } 
    else if (snsRIGHT == obj.good) {
      this.action = RIGHT;
    }
    else if ((snsFWD == obj.bad || snsFWD == obj.wall) && (snsLEFT == obj.bad || snsLEFT == obj.wall) && (snsRIGHT == obj.bad || snsRIGHT == obj.wall)) {
      this.action = RIGHT;
    }
    else if (snsFWD == obj.bad && snsLEFT == obj.bad) {
      this.action = RIGHT;
    }
    else if (snsFWD == obj.bad && snsRIGHT == obj.bad) {
      this.action = LEFT;
    }
    else if (snsLEFT == obj.bad && snsRIGHT == obj.bad) {
      this.action = FWD;
    }
    else {
      let bad_move = true;
      while (bad_move) {
      	let new_action = randint(0, 10);
        if (new_action <= 7) this.action = FWD;
        else if (new_action <= 9) this.action = LEFT;
        else this.action = RIGHT;
        bad_move = (this.sensors[this.action].val == obj.wall || this.sensors[this.action].val == obj.bad);
      }
    }
  }
  //===================================
  // NOTHING BELOW HERE SHOULD CHANGE
  //===================================

  constructor() {
    // sensor offsets [dx, dy] indexed by bot heading (0-3) and sensor number (0-2)
    //    FWD     LEFT    RIGHT 
    this.sensorOffsets = [
      [ [1, 0],  [0, -1], [0, 1]  ], // bot facing east
      [ [0, 1],  [1, 0],  [-1, 0] ], // bot facing south
      [ [-1, 0], [0, 1],  [0, -1] ], // bot facing west
      [ [0, -1], [-1, 0],  [1, 0] ] // bot facing north
    ];
    this.dia = grid.cellSize;
    this.reset();
  }

  consume() {
    
    this.reward = -0.01; // baseline cost of moving

    let objHere = grid.get(this.ix, this.iy); // object directly under the bot

    if (objHere == obj.good) {
      this.reward += 1.0;
      // move the pellet to a new location
      grid.set(this.ix, this.iy, 0);
      let cell = grid.getEmptyCell();
      grid.set(cell.ix, cell.iy, obj.good);
    } else if (objHere == obj.bad) {
      this.reward -= 1;
      // move the pellet to a new location
      grid.set(this.ix, this.iy, 0);
      let cell = grid.getEmptyCell();
      grid.set(cell.ix, cell.iy, obj.bad);
    }

    this.energy += this.reward;
  }

  display() {
    let cs = grid.cellSize;
    push();
    noStroke();
    fill(255, 255, 0, 64);
    for (let i = 0; i < this.sensors.length; i++) {
      var xc = (this.sensors[i].ix) * cs;
      var yc = (this.sensors[i].iy) * cs;
      rect(xc, yc, cs, cs);
    }
    // draw body
    stroke(0);
    // draw heading indicator
    translate((this.ix + 0.5) * cs, (this.iy + 0.5) * cs);
    fill('yellow');
    ellipse(0, 0, this.dia, this.dia);
    rotate(HALF_PI * this.heading);
    line(0, 0, this.dia / 2, 0);
    pop();
  }

  getSensorState() {
    // each sensor has 4 possible values, so need 2 bits of storage per sensor
    let state = 0;
    for (let i = 0; i < this.sensors.length; i++) {
      state += (this.sensors[i].val << 2 * i); // shift data and store
    }
    return state;
  }

  move() {
    let cmd = this.action;
    if (cmd === FWD) { // move forward if not blocked by wall
      switch (this.heading) {
        case 0: // east
          if (grid.get(this.ix + 1, this.iy) != obj.wall) {
            this.ix++;
          }
          break;
        case 1: // south
          if (grid.get(this.ix, this.iy + 1) != obj.wall) {
            this.iy++;
          }
          break;
        case 2: // west
          if (grid.get(this.ix - 1, this.iy) != obj.wall) {
            this.ix--;
          }
          break;
        case 3: // north
          if (grid.get(this.ix, this.iy - 1) != obj.wall) {
            this.iy--;
          }
          break;
      }
    } else if (cmd === LEFT) { // turn left
      this.heading--;
      if (this.heading == -1) {
        this.heading = 3;
      }

    } else if (cmd === RIGHT) { // turn right
      this.heading++;
      if (this.heading == 4) {
        this.heading = 0;
      }

    } else {
      console.log('Unrecognized action command ' + cmd);
    }
  }

  randAction() {
    // pick a random action
    this.action = randint(0, 2);
  }

  reset() {
    this.setController(select("#controller").value());
    this.ix = int(grid.nx/2);
    this.iy = int(grid.ny/2);
    this.heading = randint(0, 3); // 0-3 = east, south, west, north
    this.energy = 0;
    this.reward = 0;
    this.sensors = [{},{},{}];
    this.updateSensors();
    this.state = this.getSensorState(); // for reinforcement learning
    this.nextState = this.state; 
    this.action = undefined;
    this.nextAction = undefined;  
  }

  setController(name) {
    this.controllerName = name;
    this.controller = this[name];
  }

  Qlearner() {
    this.action = learner.actionEpsGreedy(this.state);
  }

  update() {
    this.state = this.getSensorState();
    this.controller();
    this.move();
    this.consume();
    this.updateSensors();
    this.nextState = this.getSensorState();
    if (this.controllerName == "Qlearner") {
      learner.updateQ(this.state, this.action, this.reward, this.nextState);
    }
  }

  updateSensors() {
    // update sensors
    // each sensor has 4 possible values depending on what is "under" the sensor on the grid
    // 0 = nothing, 1 = wall, 2 = good pellet, 3 = bad pellet
    let offsets = this.sensorOffsets[this.heading];
    for (var i = 0; i < this.sensors.length; i++) {
      this.sensors[i].ix = this.ix + offsets[i][0];
      this.sensors[i].iy = this.iy + offsets[i][1];
      this.sensors[i].val = grid.get(this.sensors[i].ix, this.sensors[i].iy);
    }
  }
}
