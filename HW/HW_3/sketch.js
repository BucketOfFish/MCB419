var itick = 0; // simulation time
var paused = true; // simulation run/pause status
var wrapped = false; // boundary conditions (wrapped or solid)

var bot;
var pellets = [];
const NPELLETS = 100;

function simDraw() { // visual display of simulation state
  // THIS CODE IS OK... NO CHANGES
  background(50, 50, 100);

  for (let p of pellets) p.display();
  bot.display();

  // text info (upper left corner)
  noStroke();
  fill(0);
  rect(0, 0, 100, 60); // text background
  textSize(14);
  fill('white');
  text("itick: " + itick, 10, 15); // tick count
  fill('orange')
  text(wrapped ? 'wrapped' : 'solid', 10, 30); // wrapped status 
  fill('lightGreen');
  text("collected: " + (NPELLETS - pellets.length), 10, 45);
}

function simIsDone() { // return true if done, false otherwise
  // THIS CODE IS OK... NO CHANGES
  return (itick >= 2000);
}

function simReset() { // reset simulation to initial state
  // THIS CODE IS OK... NO CHANGES
  itick = 0;
  bot.reset();
  changeNoise();
  pellets = [];
  for (let i = 0; i < NPELLETS; i++) pellets.push(new Pellet());
}

function simSetup() { // called once at beginning to setup simulation
  // THIS CODE IS OK... NO CHANGES
  createCanvas(400, 300).parent("#canvas");
  bot = new Bot();
  for (let i = 0; i < NPELLETS; i++) pellets.push(new Pellet());
  simReset();
}

function simStep() { // executes a single time step (tick)
  // THIS CODE IS OK... NO CHANGES
  itick++;
  bot.update();
  bot.consume();
}

function changeNoise() {
  // THIS CODE IS OK...NO CHANGES
  let noise = float(select("#slider_wanderNoise").value());
  select("#text_wanderNoise").html("wanderNoise = " + nf(noise, 1, 2));
  bot.wanderNoise = noise;
}

//==================================
// Nothing below here should change
// unless you add new UI elements
//==================================

function setup() {

  simSetup();

  select("#b_reset").mouseClicked(function() { // reset button
    simReset();
    paused = true;
    noLoop();
    redraw();
  });

  select("#b_run").mouseClicked(function() { // run-pause button
    paused = !paused;
    if (paused) noLoop();
    else loop();
  });

  select("#b_single").mouseClicked(function() { // single step button
    paused = true;
    noLoop();
    simStep();
    redraw();
  });

  select("#b_wrap").mouseClicked(function() { // boundary conditions
    wrapped = !wrapped;
    if (paused) redraw();
  });

  select("#slider_wanderNoise").mouseMoved(changeNoise);

  select("#b_expt").mouseClicked(runExpt);
}

function draw() {
  if (!paused) {
    simStep();
    if (simIsDone()) {
      paused = true;
      noLoop();
    }
  }
  simDraw();
}