function setup() {
    createCanvas(400, 400);
    colorMode(HSB);
}

var circles = [];
var current_circle = "NONE";
var current_color = 0;

function Circle() {

    this.size = 0;
    this.color = "red";
    this.x = 0;
    this.y = 0;
}

function draw() {

    background(220);
    current_color++;
    current_color = current_color % 360;

    if (mouseIsPressed) {
        current_circle.size++;
        current_circle.x = mouseX;
        current_circle.y = mouseY;
        current_circle.color = color(current_color, 255, 255);
    }

    for (var i=0; i<circles.length; i++) {
        fill(circles[i].color);
        ellipse(circles[i].x, circles[i].y, circles[i].size);
        if (circles[i] != current_circle) {
            circles[i].size -= 0.1;
            if (circles[i].size <= 0) {
                circles.splice(i, 1);
            }
        }
    }
}

function mousePressed() {
    current_circle = new Circle();
    circles.push(current_circle);
}

function mouseReleased() {
    current_circle = "NONE";
}
