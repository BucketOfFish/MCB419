var itick = 0; // simulation time
var paused = true; // simulation run/pause status
var validID = false;

var bots = [];
const NBOTS = 4;

var world; // the environment

var controllerNames = [
  "aggressive",
  "coward",
  "explorer",
  "love"
];

var botColors = {
  aggressive: 'red',
  coward: 'yellow',
  love: 'blue',
  explorer: 'magenta'
};

function simDraw() {
  
  if (!validID) {
    background(0);
    fill('red');
    textSize(40);
    text("Enter a valid ID in index.html", 20, height/2);
    return;
  }

  world.display(); // draws the light and scoring rings

  for (let b of bots) b.display();

  textSize(14);
  noStroke();
  fill('white');
  textAlign(LEFT);
  text("score: " + nf(world.scoring.total,0,2), 10, 15);
  textAlign(RIGHT);
  text(itick, width-10, 15);
  textAlign(CENTER);
  text(select("#controller").value(), width / 2, 15);

}

function simIsDone() { // return true if done, false otherwise
  return (world.scoring.passed || itick >= 1000);
}

function simReset() { // reset simulation to initial state
  itick = 0;
  for (let b of bots) b.reset();
  changeController();
  world.scoring.reset();
}

function simSetup() { // called once at beginning to setup simulation
  createCanvas(600, 400).parent("#canvas");
  world = new World();
  for (let i = 0; i < NBOTS; i++) {
    let r = 50 + 50 * i;
    let angle = PI / 4 + i * HALF_PI;
    let x = width / 2 + r * cos(angle);
    let y = height / 2 + r * sin(angle);
    let heading = i * HALF_PI;
    bots.push(new Bot(x, y));
  }
  // adjust light intensity
  let val = int(select("#studentID").value());
  if(val >= 0 && val <= 999){
    validID = true;
    let inten = int(10 + val);
    console.log("Light intensity: " + inten);
    world.lights[0].intensity = inten;
  }
  print("TESTING");
  print(world.lights[0].intensity);
}

function simStep() { // executes a single time step (tick)
  itick++;
  for (let b of bots) {
    b.updateSensors();
    b.controller();
    b.update();
  }
  world.scoring.update();
}

function changeController() {
  controllerName = select("#controller").value();
  for (let b of bots) {
    b.controller = b[controllerName];
    let c = botColors[controllerName];
    b.cfill = c ? c : 'grey';
    b.trail.cstroke = c ? c : 'grey';
  }
  redraw(); // to update text display of controller name
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
  controllerMenu.changed(changeController);

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