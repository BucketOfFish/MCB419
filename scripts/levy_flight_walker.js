// Levy flight random walker - P(r) ~= (1-r)^2 for r in [0,1]
// If y is uniformly distributed in [0,1], then r = 1 - (1-y)^(1/3)

function Walker() {

    this.x = width/2;
    this.y = height/2;

    this.display = function() {
        stroke(0);
        point(this.x, this.y);
    }

    this.step = function() {
        var r = 1 - Math.pow(1 - random(1), 1/3);
        length = r * 10;
        var phi = random(2 * Math.PI);
        this.x += length * Math.cos(phi);
        this.y += length * Math.sin(phi);
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
