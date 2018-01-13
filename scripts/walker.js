// Random walker that tends to go in the direction of the mouse

function Walker() {

    this.x = width/2;
    this.y = height/2;

    this.display = function() {
        stroke(0);
        point(this.x, this.y);
    }

    this.step = function() {
        var stepX = Math.sign(mouseX - this.x);
        var stepY = Math.sign(mouseY - this.y);
        if (random(1) < 0.45) stepX *= -1;
        if (random(1) < 0.45) stepY *= -1;
        if (stepX == 0) stepX = Math.sign(random(-1, 1));
        if (stepY == 0) stepY = Math.sign(random(-1, 1));
        this.x += stepX;
        this.y += stepY;
    }
}

var w;

function setup() {
    w = new Walker();
}

function draw() {
    w.step();
    w.display();
}
