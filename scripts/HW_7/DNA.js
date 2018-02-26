class DNA {

  constructor() {
    // lowerBound and upperBound are arrays, the same length as genes
    this.genes = [];
    this.lowerBound = lowerBound;
    this.upperBound = upperBound;
    this.fitness = 0;
    for (let i = 0; i < lowerBound.length; i++) {
      this.genes[i] = random(this.lowerBound[i], this.upperBound[i]);
    }
  }

  clone() {
    // returns a child DNA as an identical copy of parent
    let child = new DNA(this.genes.length);
    for (let i = 0; i < this.genes.length; i++) {
      child.genes[i] = this.genes[i];
    }
    return child;
  }

  crossover(partner) {
    // returns a child DNA from this DNA and the parent DNA
    let child = new DNA(this.genes.length);

    let xpoint = floor(random(this.genes.length));

    for (let i = 0; i < this.genes.length; i++) {
      if (i > xpoint) child.genes[i] = this.genes[i];
      else child.genes[i] = partner.genes[i];
    }
    return child;
  }

  mutate() {
    // mutate a single gene
    let i = floor(random(this.genes.length));
    this.genes[i] *= random(0.8, 1.2); // local search
    this.genes[i] = constrain(this.genes[i], this.lowerBound[i], this.upperBound[i]);
  }
}