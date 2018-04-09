// simulation globals
var itick;
var bot;
var attractant;
var trail

const NSTEPS = 1000;
const NTRIALS = 10;

// GA globals - values assigned in "ga_specs" input field
var popSize;
var crossoverRate;
var mutationRate;
var lowerBound; // lower bound on gene values (an array, same length as genes)
var upperBound; // upper bound on gene values (an array, same length as genes)

var graph; // graphical output
var population; // population

var wallStart, wallElapsed = 0;
var paused = true;
var evolving = false;

function calcFitness(genes) {
  bot.genes = genes.slice();
  let peakVal = [0.1, 1.0, 10.0];
  let fitsum = 0;
  for (let ip = 0; ip < peakVal.length; ip++) {
    attractant.peak = peakVal[ip];
    itick = 0;
    bot.reset();
    for (let i = 0; i < NSTEPS; i++) {
      itick++;
      bot.updateSensors();
      bot.controller();
      bot.update();
    }
    fitsum += bot.fitness;
  }
  return fitsum;
}

function diffAngle(a, b) {
  var TWOPI = 2 * Math.PI;
  var ar = (a + Math.PI) % TWOPI - Math.PI;
  var br = (b + Math.PI) % TWOPI - Math.PI;
  var delta = ar - br;
  if (delta > Math.PI) delta -= TWOPI;
  if (delta < -Math.PI) delta += TWOPI
  return delta;
}

function finished() {
  // specifies the termination criteria
  return population.generation >= 100;
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
    outp.html("genes: " + toString(bot.genes, 2) + "<br><br>");
    outp.html(" 0.1   1.0   10.0 |  sum <br>", true);
    outp.html("----- ----- ----- | ------<br>", true);

    let peakVal = [0.1, 1.0, 10.0];
    let fitsum = 0;
    let fitmax = -Infinity;
    for (let itrial = 0; itrial < NTRIALS; itrial++) {
      let fit3 = 0; // fitness sum for 3 different peak values
      for (let ip = 0; ip < peakVal.length; ip++) {
        attractant.peak = peakVal[ip];
        simReset();
        for (let i = 0; i < NSTEPS; i++) simStep();
        outp.html(nf(bot.fitness, 3, 1) + " ", true);
        fit3 += bot.fitness;
      }
      outp.html("| " + nf(fit3, 3, 1) + "<br>", true);
      fitsum += fit3;
      if (fit3 > fitmax) fitmax = fit3;
    }
    fitsum /= NTRIALS
    outp.html("<br>Fitness Avg: " +
      nf(fitsum, 0, 1) +
      " Best: " +
      nf(fitmax, 0, 1) +
      "<br>", true);
    attractant.peak = 1;
    select("#peak").value(attractant.peak);
    redraw();
  });

  select("#peak").value(attractant.peak);
  select("#peak").changed(function() {
    attractant.peak = float(select("#peak").value());
    select("#peak").value(attractant.peak);
    console.log("attractant peak = " + attractant.peak);
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
    select("#specs").value("bot.genes = " + toString(best, 2));
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