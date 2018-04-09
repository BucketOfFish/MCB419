class Neuron_IAF {
  constructor(name, x, y) {
    this.name = name;
    this.x = x;
    this.y = y;
    this.vm = 0;
    this.thresh = 10;
    this.output = 0;
    this.dia = 15;
    this.fillColor = color(50, 150, 50);
    this.incoming = [];
    this.outgoing = [];
  }

  update() {
    // integrate-and-fire rules
    if (this.output > 0.0) {
      this.reset();
    } else {
      //this.vm += this.summed_inputs();
      for (let i = 0; i < this.incoming.length; i++) {
        this.vm += this.incoming[i].weight * this.incoming[i].pre.output;
      }
      if (this.vm > this.thresh) this.spike();
    }
  }

  summed_inputs() {
    // weighted-sum of incoming synaptic connections
    var isum = 0;
    for (var i = 0; i < this.incoming.length; i++) {
      isum += this.incoming[i].weight * this.incoming[i].pre.output;
    }
    return isum;
  }

  reset() {
    this.vm = 0;
    this.output = 0;
  }

  spike() {
    this.vm = 5 * this.thresh; // increase vm so spikes stand out on vm graph
    this.output = 1.0; // 1.0 = spike, 0.0 = no spike
  }

  display() {
    if (this.output > 0.0) {
      fill(255); // change color to white when spiking
    } else {
      fill(this.fillColor);
    }
    noStroke();
    ellipse(this.x, this.y, this.dia, this.dia);
    if (showLabels) {
      push();
      textAlign(CENTER, CENTER);
      textSize(10);
      fill(this.fillColor);
      text(this.name, this.x, this.y + 12);
      pop();
    }
  }
}