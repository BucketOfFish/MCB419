class Axon {
  constructor(preName, postName, weight) {
    this.pre = null; // pre-synaptic unit
    this.post = null; // post-synaptic unit
    this.weight = weight;
    // find pre- and post-synaptic units in the array by name
    for (let i = 0; i < units.length; i++) {
      if (units[i].name === preName) this.pre = units[i];
      if (units[i].name === postName) this.post = units[i];
    }
    if (this.pre === null) alert("presynaptic unit named " + preName + " not found!");
    if (this.post === null) alert("postsynaptic unit named " + postName + " not found!");

    // add this axon to incoming and outgoing connection lists
    this.pre.outgoing.push(this);
    this.post.incoming.push(this);
  }

  display() {
    // draw lines between units
    stroke(250, 250, 0, 128);
    line(this.pre.x, this.pre.y, this.post.x, this.post.y);
  }
}