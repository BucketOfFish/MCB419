class Population {
  
  constructor(popSize) {
    this.generation = 0;
    this.pool = [];
    for (let i = 0; i < popSize; i++) {
      this.pool[i] = new DNA(lowerBound, upperBound);
    }
  }
  
  selectAndReproduce() {
    var newPool = [];
    var child = null;
    
    for (let i = 0; i < this.pool.length; i++) {
      let dad = this.select1();
      if (Math.random() < crossoverRate) {
        let mom = this.select1();
        child = dad.crossover(mom);
      } else {
        child = dad.clone();
      }
      if (Math.random() < mutationRate) child.mutate();
      newPool.push(child);
    }
    this.pool = newPool;
    this.generation++;
  }

  evaluateFitness() {
    // evaluate fitness of all individuals
    for (let i = 0; i < this.pool.length; i++) {
      this.pool[i].fitness = calcFitness(this.pool[i].genes);
    }
    // sort the results, best (largest) first
    this.pool.sort((a, b) => b.fitness - a.fitness);
  }

  //------------------------------------------
  // SELECTION methods
  //------------------------------------------
  select1() {
    // tournament selection, size = 2
    // pick two random individuals from the pool 
    // and return the best one
    let n = this.pool.length;
    let a = this.pool[floor(random(n))];
    let b = this.pool[floor(random(n))];
    return (a.fitness > b.fitness) ? a : b;
  }

  //------------------------------------------
  // GET methods - no computational role in GA
  //------------------------------------------

  // displayAll
  displayAll() {
    let everything = "";
    let displayLimit = min(this.pool.length, 50);
    for (let i = 0; i < displayLimit; i++) {
      everything += toString(this.pool[i].genes, 2) + " | " + this.pool[i].fitness + "<br>";
    }
    return everything;
  }

  // getAverageFitness
  getAverageFitness() {
    let total = 0;
    for (let i = 0; i < this.pool.length; i++) {
      total += this.pool[i].fitness;
    }
    return total / (this.pool.length);
  }

  // getBestFitness
  getBestFitness() {
    return this.pool[0].fitness;
  }

  // getBest
  getBestDNA() {
    return this.pool[0];
  }

  // getGeneration
  getGeneration() {
    return this.generation;
  }
}