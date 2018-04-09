var itick = 0; // simulation time
var paused = true; // simulation run/pause status

var bot;
var grid;
var learner;

const NSTEPS = 2000;

const controllerNames = ['randAction', 'handCoded', 'Qlearner'];
const obj = {wall:1, good:2, bad:3}; 

const FWD = 0;
const LEFT = 1;
const RIGHT = 2;

function displayQ() {
  let div = select("#dispQ");
  
  // print header
  div.html('Q[state]: [FWD, LEFT, RIGHT]\n');
  for (let state of [0, 2, 3, 8, 20, 32]) {
    let qval = learner.Q[state];
    div.html(sprintf("Q[%d]: [%4.3f, %4.3f, %4.3f]\n", 
      state, qval[0], qval[1], qval[2] ), true);
  }
}

function setupGrid() {
	var nx = 30;
	var ny = 20;
	var cellSize = 20;
	grid = new Grid(nx, ny, cellSize);

	// create top/bottom walls
	for (let i = 0; i < grid.nx; i++) {
		grid.set(i, 0, obj.wall);
		grid.set(i, grid.ny - 1, obj.wall);
	}
	// create left/right walls
	for (let j = 0; j < grid.ny; j++) {
		grid.set(0, j, obj.wall);
		grid.set(grid.nx - 1, j, obj.wall);
	}

	// create pellets
	for (let i = 0; i < 15; i++) {
		let cell  = grid.getEmptyCell();
		grid.set(cell.ix, cell.iy, obj.good);
		cell = grid.getEmptyCell();
		grid.set(cell.ix, cell.iy, obj.bad);
	}
}


function simDraw() {
  background(0);
	grid.display();
	bot.display();
  
  // display info
	let cs = grid.cellSize;
	push();
	textSize(14);
	noStroke();
	fill('white');
  textAlign(LEFT);
  text("state = " + bot.nextState, 1.5*cs, 1.8*cs);
	textAlign(CENTER);
	text("energy = " + nf(bot.energy,1,2), width / 2, 1.8*cs);
	textAlign(RIGHT);
	text(itick, width - 1.5*cs, 1.8*cs);
	pop();
}

function simIsDone() { // return true if done, false otherwise
  return (itick >= NSTEPS);
}

function simReset() { // reset simulation to initial state
  itick = 0;
  paused = true;
  // clear old pellets
	for (let i = 0; i < grid.nx; i++) {
		for (let j = 0; j < grid.ny; j++) {
			if(grid.get(i,j) > obj.wall) {grid.set(i,j,0);}
		}
	}

	// add new pellets
	for (let i = 0; i < 15; i++) {
		let cell = grid.getEmptyCell();
		grid.set(cell.ix, cell.iy, obj.good);
		cell = grid.getEmptyCell();
		grid.set(cell.ix, cell.iy, obj.bad);
	}
  
  bot.reset();
}

function simSetup() { // called once at beginning to setup simulation
  setupGrid();
  createCanvas(grid.nx*grid.cellSize, grid.ny*grid.cellSize).parent('#canvas');
  bot = new Bot();
  learner = new Qlearner(64, 3);
  simReset();
}

function simStep() { // executes a single time step (tick)
  itick++;
  bot.update();
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
    redraw();
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
  
  select("#b_resetQ").mouseClicked(function() { // resetQ button
    learner.resetQ();
  });
  
  select("#b_dispQ").mouseClicked(function() { // dispQ button
    displayQ();
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