// Randomly distributed heights in a population

function setup() {
    createCanvas(640, 360);
}

function draw() {
    var z = randomGaussian();
    var std = 60;
    var mean = 320;
    var height = z*std+mean;
    noStroke(); // no border
    fill(0, 10); // black, but almost transparent
    ellipse(height, 180, 16, 16); // x, y, sizeX, sizeY
}
