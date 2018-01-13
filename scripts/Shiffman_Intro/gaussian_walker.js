// Gaussian random walker

function Walker() {

    this.x = width/2;
    this.y = height/2;

    this.display = function() {
        stroke(0);
        point(this.x, this.y);
    }

    this.step = function() {
        var stepX = randomGaussian();
        var stepY = randomGaussian();
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
