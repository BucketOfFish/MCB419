// THIS CODE IS OK... NO CHANGES IN THIS FILE

function runExpt() {
  var stats;
  var ival = 0;
  var itrial = 0;
  var collected = [];
  var rawdiv; // HTML <div> for raw data
  var stastsdiv; // HTML <div> for summary statistics
  var wanderNoise;

  // evaluate experiment design specifications
  let specs = select("#specs").value();
  eval(specs);

  // this is the first trial of the experiment
  wanderNoise = noiseValues[0];
  select("#slider_wanderNoise").value(wanderNoise);
  changeNoise();
  
  // print out headers
  rawdiv = select('#rawdata');
  let bc = wrapped ? 'WRAPPED\n' : 'SOLID\n'
  rawdiv.html('Boundary conditions: ' + bc + '\n');
  rawdiv.html('noise | # pellets collected\n', true);
  rawdiv.html('------+--------------------\n', true);
  rawdiv.html(sprintf(" %3.2f | ", wanderNoise), true);

  statsdiv = select('#stats');
  statsdiv.html('noise | mean +- err\n');
  statsdiv.html('------+------------\n', true);
  nextTrial();
  
  function nextTrial() {
    // run a single trial
    simReset();
    for (let istep = 0; istep < nsteps; istep++) {
      simStep();
    }
    let ncollected = NPELLETS - pellets.length;
    collected.push(ncollected);
    rawdiv.html(sprintf("%3d", ncollected), true);

    itrial++;
    if (itrial < ntrials) {
      setTimeout(nextTrial, 10); // run another trial
    } else {
      // finished ntrials; calc stats and put in table
      stats = calcArrayStats(collected);
      var mean = nf(stats.mean, 0, 2);
      var sem = nf(stats.sem, 0, 2);
      var output = mean + " &plusmn; " + sem;
      statsdiv.html(sprintf(" %3.2f | %4.1f +- %2.1f\n", wanderNoise, mean, sem), true);
      // get next noise value
      nextValue();
    }
  }

  function nextValue() {
    ival++;
    if (ival < noiseValues.length) {
      // start a new set of trials
      itrial = 0;
      collected = [];
      wanderNoise = noiseValues[ival];
      select("#slider_wanderNoise").value(wanderNoise);
      changeNoise();
      rawdiv.html(sprintf("\n %3.2f | ", wanderNoise), true);
      setTimeout(nextTrial, 10); // start the next trial
    }
  }
}

function calcArrayStats(inputArray) {
  //
  // calculates mean, standard deviation and standar error of Array elements 
  //
  // input: inputArray, an array of numbers
  // returns: {mean: <mean>, std: <standard deviation>, sem: <standard error>}
  //
  let sum = 0;
  let sumSq = 0;
  let n = inputArray.length;
  for (let i = 0; i < n; i++) {
    sum += inputArray[i];
    sumSq += inputArray[i] * inputArray[i];
  }
  let variance = (sumSq - (sum * sum) / n) / (n - 1);
  return {
    mean: sum / n,
    std: Math.sqrt(variance),
    sem: Math.sqrt(variance / n)
  };
}