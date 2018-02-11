function runExpt() {

  let statsdiv = select("#stats");
  let cselect = select("#controller");
  
  // print header
  statsdiv.html('     controller | mean +- err\n');
  statsdiv.html('----------------+---------------\n', true);
  
  // loop over controllers
  for (let i = 0; i < controllerNames.length; i++) {
    let cname = controllerNames[i];
    cselect.value(cname);
    statsdiv.html(sprintf("%15s |", cname), true);
    
    // run the simulation
    simReset();
    for (let i=0; i<2000; i++) simStep();
    
    // fill an array with temperature data
    let tempArray = []; // array of temperatures, one entry per bot
    for (let b of bots)tempArray.push(world.getTemperature(b.x, b.y));
    
    // calculate and print statistics
    let stats = calcArrayStats(tempArray);
    statsdiv.html(sprintf(" %3.2f +- %3.3f\n", stats.mean, stats.sem), true);
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