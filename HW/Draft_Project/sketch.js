function setup() {
	createCanvas(400, 300).parent('canvas');
	createGUI();
	background('darkBlue');
	fill('orange');
	textAlign(CENTER);
	textSize(24);
	text("MCB 419 Project DRAFT", width/2, height/2);
	noLoop();
}

function createGUI() {
	var btn = createButton('click me').parent('gui');
	btn.mousePressed(function() {
		background(random(255), random(255), random(255));
    fill(0);
    text("MCB 419 Project DRAFT", width/2, height/2);
		redraw();
	});
}