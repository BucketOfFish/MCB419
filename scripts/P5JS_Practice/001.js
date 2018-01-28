function setup() {
    createCanvas(400, 400);
}

function draw() {

    background(220);

    noStroke();
    fill('white');
    quad(
        175, 200, // x (from left), y (from top)
        250, 175,
        250, 325,
        175, 350
    );
    quad(
        175, 200,
        175, 350,
        140, 310,
        140, 180
    );

    fill('brown');
    quad(
        175, 300,
        250, 275,
        250, 325,
        175, 350
    );
    quad(
        175, 300,
        175, 350,
        140, 310,
        140, 260
    );

    fill('red');
    quad(
        175, 240,
        250, 225,
        250, 300,
        175, 325
    );
    quad(
        175, 240,
        175, 325,
        150, 300,
        150, 230
    );

    fill('white');
    beginShape();
        vertex(140, 180);
        vertex(175, 200);
        vertex(250, 175);
        vertex(210, 150);
        vertex(175, 150);
    endShape(CLOSE);

    fill('black');
    beginShape();
        vertex(210, 150);
        vertex(185, 155);
        vertex(175, 150);
        vertex(180, 100);
        vertex(205, 100);
    endShape(CLOSE);
    quad(
        177, 100,
        208, 100,
        208, 78,
        177, 78
    );
}
