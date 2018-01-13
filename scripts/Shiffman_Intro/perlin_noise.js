// Perlin noise

var timeX = 0;
var timeY = 10000;
var stepSize = 5;

function Walker() {

    this.x = width/2;
    this.y = height/2;

    this.display = function() {
        clear();
        ellipse(this.x, this.y, 16, 16); // x, y, sizeX, sizeY
    }

    this.step = function() {
        var protoX = noise(timeX); // Perlin noise at a time step
        var protoY = noise(timeY);
        timeX += 0.01;
        timeY += 0.01;
        var stepX = map(protoX, 0, 1, -stepSize, stepSize); // value, oldLow, oldHigh, newLow, newHigh
        var stepY = map(protoY, 0, 1, -stepSize, stepSize);
        this.x += stepX;
        this.y += stepY;
    }
}

var w;

function setup() {
    createCanvas(1000, 1000);
    w = new Walker();
}

function draw() {
    w.step();
    w.display();
}
