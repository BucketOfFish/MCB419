var itick = 0; // simulation time
var paused = true; // simulation run/pause status

var bot;
var trail;
var pellets = [];

const NSTEPS = 2000;

// order of pellets from best to worst
// this order will be randomized during setup
var valueOrder = ['red', 'green', 'blue'];

var controllerNames = [
  "seekUser",
  "seekAll",
  "seekRed",
  "seekGreen",
  "seekBlue"
];

function randomizeValueOrder() {
  // random permutation to determine value order (BEST, MIDDLE, WORST)
  for (var i = 2; i >= 0; i--) {
    var randIdx = floor(random(i + 1));
    var tmp = valueOrder[i];
    valueOrder[i] = valueOrder[randIdx];
    valueOrder[randIdx] = tmp;
  }
}

function setupPellets() {
  pellets = [];
  for (let ival = 0; ival < 3; ival++) {
    let pColor = valueOrder[ival];
    for (let i = 0; i < 10; i++) {
      pellets.push(new Pellet(pColor));
    }
  }
}


function simDraw() {
  background(200);
  bot.display();
  trail.display();
  for (let p of pellets) p.display();

  push();
  textSize(14);
  noStroke();
  
  fill(220);
  rect(0, 0, 110, 70);
  textAlign(LEFT);
  fill('black');
  text("Energy = " + bot.energy, 5, 15);
  fill('red');
  text("Est red: " + nf(bot.estimatedValue[0],1,2), 5, 30);
  fill('green');
  text("Est green: " + nf(bot.estimatedValue[1], 1, 2), 5, 45);
  fill('blue');
  text("Est blue: " + nf(bot.estimatedValue[2], 1, 2), 5, 60);
  
  textAlign(CENTER);
  fill('white');
  text(bot.controllerName, width / 2, 15);
  
  textAlign(RIGHT);
  text(itick, width - 5, 15);
  pop();

}

function simIsDone() { // return true if done, false otherwise
  return (itick >= NSTEPS);
}

function simReset() { // reset simulation to initial state
  itick = 0;
  paused = true;
  bot.reset();
  trail.reset();
  setupPellets();
}

function simSetup() { // called once at beginning to setup simulation
  createCanvas(600, 400).parent("#canvas");
  bot = new Bot();
  trail = new Trail(NSTEPS, 'orange');
  randomizeValueOrder();  // different "best" pellet for each setup
  simReset();
}

function simStep() { // executes a single time step (tick)
  itick++;
  bot.update();
  trail.update(bot.x, bot.y, bot.sns.lastColorConsumed);
}

//==================================
// Nothing below here should change
// unless you add new UI elements
//==================================

function setup() {

  simSetup();

  // controller select
  let controllerMenu = select("#controller");
  for (let i = 0; i < controllerNames.length; i++) {
    controllerMenu.option(controllerNames[i]);
  }
  controllerMenu.changed(function() {
    let cname = select("#controller").value();
    bot.setController(cname);
  });

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

  select("#b_expt").mouseClicked(runExpt);

  simReset();
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