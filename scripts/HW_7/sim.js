function simDraw() { // visual display of simulation state
  background(0);
  textSize(16);
  fill('white');
  textAlign(LEFT);
  text("tick: " + itick, 10, 20);
  textAlign(RIGHT);
  text("fit: " + nf(bot.fitness, 0, 1), width - 10, 20);

  // attractant
  push();
  noStroke();
  translate(attractant.x, attractant.y);
  var denom = 2 * attractant.sigma * attractant.sigma;
  for (var r = 200; r > 0; r -= 4) {
    var alpha = attractant.peak * Math.exp(-(r * r) / denom);
    fill(100, 64, 0, 128 * alpha);
    ellipse(0, 0, 2 * r);
  }
  pop();
  bot.trail.display();
  bot.display();
}

function simIsDone() { // return true if done, false otherwise
  return (itick >= NSTEPS);
}

function simReset() { // reset simulation to initial state
  itick = 0;
  bot.reset();
  bot.trail.reset();
  let specs = select("#specs").value();
  eval(specs);
}

function simSetup() { // called once at beginning to setup simulation
  createCanvas(400, 400).parent("#canvas");
  attractant = {
    x: width / 2,
    y: height / 2,
    peak: 1.0,
    sigma: height / 4
  };
  bot = new Bot();
  bot.trail = new Trail(1000, 'grey');
  simReset();
  noLoop();
}

function simStep() { // executes a single time step (tick)
  itick++;
  bot.updateSensors();
  bot.controller();
  bot.update();
  bot.trail.update(bot.x, bot.y);
}