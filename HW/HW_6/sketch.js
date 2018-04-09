// simulation globals
var itick;
var bot;
var pellets = [];
const NPELLETS = 20;
const NSTEPS = 2000;
const NTRIALS = 100;

// GA globals - values assigned in "ga_specs" input field
var popSize;
var crossoverRate;
var mutationRate;
var lowerBound;  // lower bound on gene values (an array, same length as genes)
var upperBound;  // upper bound on gene values (an array, same length as genes)

var graph; // graphical output
var population; // population

var wallStart, wallElapsed = 0;
var paused = true;
var evolving = false;

function calcFitness(genes) {
  simReset();
  bot.genes = genes;
  for (let i = 0; i < NSTEPS; i++) simStep();
  return bot.fitness;
}

function finished() {
  // specifies the termination criteria
  return population.generation >= 50;
}

function setup() {
  createCanvas(400, 300).parent("canvas");
  graph = createGraph(["best", "avg"], {
    stepHorizon: 50
  });
  simSetup();
  select("#b_reset").mouseClicked(function() { // reset button
    simReset();
    evolving = false;
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

  select("#b_expt").mouseClicked(function() { // single step button
    paused = true;
    noLoop();
    simReset();
    let outp = select("#outp");
    outp.html(""); // clear outp
    outp.html("genes: " + bot.genes.toString(2) + "<br>");
    outp.html("trials:<br> ", true);
    
    let fitsum = 0;
    let fitmax = -Infinity;
    for (let itrial = 0; itrial < NTRIALS; itrial++) {
      simReset();
      for (let i = 0; i < NSTEPS; i++) simStep();
      outp.html(bot.fitness + " ", true);
      if (itrial % 10 == 9)outp.html("<br>", true);
      fitsum += bot.fitness;
      if(bot.fitness > fitmax) fitmax = bot.fitness;
    }
    outp.html("<br>Mean fitness: " + nf(fitsum / NTRIALS, 0, 1), true);
    outp.html("<br>Max  fitness: " + fitmax, true);
    redraw();
  });

  select("#ga_reset").mouseClicked(function() {
    wallStart = millis() / 1000;

    // evaluate GA specifications
    let ga_specs = select("#ga_specs").value();
    eval(ga_specs);

    population = new Population(popSize);
    graph.reset();
    population.evaluateFitness();

    evolving = true;
    paused = true;
    noLoop();
    redraw();
  });

  select("#ga_run").mouseClicked(togglePaused);
  select("#ga_load").mouseClicked(function() {
    let best = population.pool[0].genes;
    select("#specs").value("bot.genes = " + toString(best,2));
    bot.genes = best.slice();
  });
}


function draw() {

  if (evolving) {
    graph.draw("fig1");
    displayInfo();
    if (paused) return;
    // Update GA population
    population.selectAndReproduce();
    population.evaluateFitness();
    wallElapsed = (millis() / 1000) - wallStart;
    graph.add(
      population.getGeneration(), [population.getBestFitness(), population.getAverageFitness()]);

    if (finished()) {
      paused = true;
      redraw();
      noLoop();
    }
  } else {
    if (!paused) {
      simStep();
      if (simIsDone()) {
        paused = true;
        noLoop();
      }
    }
    simDraw();
  }
}

function displayInfo() {
  // Display current status of populationation
  select('#best').html("Best gene: " + toString(population.getBestDNA().genes, 2));
  select('#best').html("<br>Best fit:   " + nf(population.getBestFitness(), 0, 2), 1);
  select('#best').html("<br>Avg fit:   " + nf(population.getAverageFitness(), 0, 2), 1);

  let statstext = "total generations:     " + population.getGeneration() + "<br>";
  statstext += "maximum fitness:       " + nf(population.getBestFitness(), 1, 2) + "<br>";
  statstext += "average fitness:       " + nf(population.getAverageFitness(), 1, 2) + "<br>";
  statstext += "elapsed time:          " + nf(wallElapsed, 1, 2) + " sec";

  select('#stats').html(statstext);
  //select('#all').html(population.displayAll());
}

function togglePaused() {
  paused = !paused;
  if (paused) {
    noLoop();
  } else {
    loop();
  }
}

// convert gene array to string representation for printing
// with nsig digits after the decimal point
function toString(genes, nsig) {
  let astr = "[";
  for (let i = 0; i < genes.length; i++) {
    astr += nf(genes[i], 0, nsig) + ", ";
  }
  astr = astr.slice(0, -2); // remove last 2 characters
  astr += "]";
  return astr;
}