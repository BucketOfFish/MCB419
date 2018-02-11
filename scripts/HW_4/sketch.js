var itick = 0; // simulation time
var paused = true; // simulation run/pause status

var faster = 3; // number of simulation steps per screen update

var bots = [];
const NBOTS = 100;

var world; // the environment (provides temperature info)

var controllerNames = [
  "wander",
  "orthokinesisPos",
  "orthokinesisNeg",
  "runTumble",
  "klinokinesisPos",
  "klinokinesisNeg"
];

function simDraw() {
  world.display(); // draws the background image

  for (let b of bots) b.display();

  // text info (upper left corner)
  fill(255);
  textSize(16);
  // display controller name in upper left corner
  textAlign(LEFT);
  var info = select("#controller").value();
  text(info, 5, 15);

  // display the frameCount in upper right corner
  textAlign(RIGHT);
  text(itick, width - 5, 15);

  // compute mean temp of bots
  textAlign(CENTER);
  var meanTemp = 0;
  for (let b of bots) meanTemp += world.getTemperature(b.x, b.y);
  meanTemp /= bots.length;
  text('mean Temp = ' + nf(meanTemp, 1, 2), width / 2, 15);
}

function simIsDone() { // return true if done, false otherwise
  return (itick >= 2000);
}

function simReset() { // reset simulation to initial state
  itick = 0;
  for (let b of bots) b.reset();
  changeController();
}

function simSetup() { // called once at beginning to setup simulation
  createCanvas(400, 300).parent("#canvas");
  world = new World();
  for (let i = 0; i < NBOTS; i++) bots.push(new Bot());
}

function simStep() { // executes a single time step (tick)
  itick++;
  for (let b of bots) {
    b.controller();
    b.update();
  }
}

function changeController() {
  controllerName = select("#controller").value();
  for (let b of bots) b.controller = b[controllerName];
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
  
  select("#faster").value(faster);
  select("#faster").changed(function() {
    faster = int(select("#faster").value());
    select("#faster").value(faster);
    console.log("faster = " + faster);
  });

  select("#b_expt").mouseClicked(runExpt);
  
  simReset();
}

function draw() {
  if (!paused) {
    // execute multiple simulation steps before drawing
    for (let i = 0; i < faster; i++) {
      simStep();
      if (simIsDone()) {
        paused = true;
        noLoop();
        break;
      }
    }
  }
  simDraw();
}