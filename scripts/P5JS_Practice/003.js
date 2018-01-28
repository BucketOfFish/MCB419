function Ball() {

    this.x = 200;
    this.y = 200;
    this.size = 10;
    this.x_vel = 0;
    this.y_vel = 0;
    this.x_acc = 0;
    this.y_acc = 0;
}

var ball;
var gravity = 0.5;
var x_grav = 0;
var y_grav = 0;
var bounce_frac = 0.5

function setup() {
    ball = new Ball();
    createCanvas(400, 400).parent("#canvas");
    select("#left").mouseClicked(function() {
        x_grav = -gravity;
        y_grav = 0;
    });
    select("#right").mouseClicked(function() {
        x_grav = gravity;
        y_grav = 0;
    });
    select("#up").mouseClicked(function() {
        x_grav = 0;
        y_grav = -gravity;
    });
    select("#down").mouseClicked(function() {
        x_grav = 0;
        y_grav = gravity;
    });
}

function draw() {
    background('lightBlue');
    fill('red');
    ellipse(ball.x, ball.y, ball.size);

    ball.x += ball.x_vel;
    ball.y += ball.y_vel;
    ball.x_acc = x_grav;
    ball.y_acc = y_grav;
    ball.x_vel += ball.x_acc;
    ball.y_vel += ball.y_acc;

    if (ball.x >= width) {
        ball.x_vel = -ball.x_vel * bounce_frac;
        ball.x = width;
    }
    if (ball.x <= 0) {
        ball.x_vel = -ball.x_vel * bounce_frac;
        ball.x = 0;
    }
    if (ball.y >= height) {
        ball.y_vel = -ball.y_vel * bounce_frac;
        ball.y = height;
    }
    if (ball.y <= 0) {
        ball.y_vel = -ball.y_vel * bounce_frac;
        ball.y = 0;
    }
}
