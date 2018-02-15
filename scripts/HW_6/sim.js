function simDraw() { // visual display of simulation state
  background('lightBlue');
  textSize(16);
  fill('blue');
  textAlign(LEFT);
  text("tick: " + itick, 10, 20);
  textAlign(RIGHT);
  text("fit: " + bot.fitness, width - 10, 20);

  for (let p of pellets) p.display();
  bot.display();
}

function simIsDone() { // return true if done, false otherwise
  return (itick >= NSTEPS);
}

function simReset() { // reset simulation to initial state
  itick = 0;
  bot.reset();
  for (let p of pellets) p.reset();
  let specs = select("#specs").value();
  eval(specs);
}

function simSetup() { // called once at beginning to setup simulation
  createCanvas(400, 300).parent("#canvas");
  bot = new Bot();
  for (let i = 0; i < NPELLETS; i++) pellets.push(new Pellet());
  simReset();
}

function simStep() { // executes a single time step (tick)
  itick++;
  bot.updateSensors();
  bot.update();
  bot.consume();
}