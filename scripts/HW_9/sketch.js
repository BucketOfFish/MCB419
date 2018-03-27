// Week 9 Homework, Lamprey place-code to rate-code conversion
// 
// Neural circuits for place-code to rate-code conversion 
// from Fig. 3 of the paper by Kamali-Sarvestani et al. (2012).

var showLabels = true;
var preyWiring = true;

var light;
var axons = [];
var units = [];

//-----------
//  SETUP
//-----------

function setup() {
	
	createCanvas(640, 480).parent('#canvas');
	createGUI();

	// light source (light is draggable)
	light = new Light(100, 100, 1000);

	// NOTE: add units in the order of update:
	// sensors, then receptor neurons, then auxillary neurons, then motors
	// start indexing at 1 to agree with Kamali-Sarvestani et al. (2012)
	for (let i = 1; i <= 8; i++) { // sensor layer
		let xoffset = 80 + 24 * i;
		units.push(new LightSensor("SL" + i, 80 + 24 * i, 150));
		units.push(new LightSensor("SR" + i, 520 - 24 * i, 150));
	}

	for (let i = 1; i <= 8; i++) { // response layer
		let neuron = new Neuron_IAF("RL" + i,  80 + 24 * i, 210);
		neuron.fillColor = color(50, 150, 50); // green
		units.push(neuron);
		neuron = new Neuron_IAF("RR" + i, 520 - 24 * i, 210);
		neuron.fillColor = color(50, 150, 50); // green
		units.push(neuron);
	}

	for (let i = 1; i <= 8; i++) { // auxillary layer
		let neuron = new Neuron_IAF("AL" + i, 80 + 24 * i, 280);
		neuron.fillColor = color(50, 150, 150); // cyan
		units.push(neuron);
		neuron = new Neuron_IAF("AR" + i, 520 - 24 * i, 280);
		neuron.fillColor = color(50, 150, 150); // cyan
		units.push(neuron);
	}

	// motors
	units.push(new Motor("ML", 80 + 24 * 4.5, 380));
	units.push(new Motor("MR", 520 - 24 * 4.5, 380));

	// connections
	makeConnections();
}

function makeConnections() {
	clearConnections();
	if (preyWiring) {
		add_connections_for_prey_approach();
	} else {
		add_connections_for_predator_escape();
	}
}

function clearConnections() {
	axons = [];
	for (let i = 0; i < units.length; i++) {
		units[i].incoming = [];
		units[i].outgoing = [];
	}
}

function add_connections_for_prey_approach() {

	// sensors to response layer
	for (let i = 1; i <= 8; i++) {
		axons.push(new Axon("SL" + i, "RL" + i, 2.0));
		axons.push(new Axon("SR" + i, "RR" + i, 2.0));
	}

	// response to auxillary layer
	for (let i = 1; i <= 8; i++) {
        for (let j = i; j <= 8; j++) {
                axons.push(new Axon("RL" + i, "AL" + j, 20));
                axons.push(new Axon("RR" + i, "AR" + j, 20));
        }
	}

	// auxillary layer to motors
	for (let i = 1; i <= 8; i++) {
		axons.push(new Axon("AL" + i, "MR", 0.004));
		axons.push(new Axon("AR" + i, "ML", 0.004));
	}
}

function add_connections_for_predator_escape() {

	// sensors to response layer
	for (let i = 1; i <= 8; i++) {
		axons.push(new Axon("SL" + i, "RL" + (9-i), 2.0));
		axons.push(new Axon("SR" + i, "RR" + (9-i), 2.0));
	}

	// response to auxillary layer
	for (let i = 1; i <= 8; i++) {
        for (let j = i; j <= 8; j++) {
                axons.push(new Axon("RL" + i, "AL" + j, 20));
                axons.push(new Axon("RR" + i, "AR" + j, 20));
        }
	}

	// auxillary layer to motors
	for (let i = 1; i <= 8; i++) {
		axons.push(new Axon("AL" + i, "ML", 0.004));
		axons.push(new Axon("AR" + i, "MR", 0.004));
	}
}

//--------------
// MAIN LOOP 
//--------------

function draw() {
	background(0);
	drawText();
	light.display();
	for (let i = 0; i < axons.length; i++) {
		axons[i].display();
	}
	for (let i = 0; i < units.length; i++) {
		units[i].update();
		units[i].display();
	}
}

function drawText() {
  push();
  noStroke();
  textAlign(CENTER, CENTER);
  // TOP TEXT
  fill(250);
  textSize(18);
  text("Lamprey place-code to rate-code conversion", width/2, 15);
  fill(250, 250, 0);
  textSize(12);
  text("click and drag the light source to move it to a new location", width/2, 35);
  textSize(16);
  fill(250, 50, 250);
  var modeStr = (preyWiring === true) ? "PREY APPROACH WIRING" : "PREDATOR ESCAPE WIRING";
  text(modeStr, width/2, 55);

  // LAYER LABELS
  fill(250);
  textSize(12);
  var xoff = 50;
  text("input", xoff, 150);
  text("response", xoff, 210);
  text("auxillary", xoff, 280);
  text("motors", xoff, 380);

  // BOTTOM TEXT
 
  textSize(12);
  fill(250);
  text("Based on Fig. 3 of Kamali-Sarvestani et al. 2012", 300, 455);

  pop();
}

//--------------
// USER INTERFACE
//--------------

function createGUI() {
	var checkLabels = createCheckbox('Show Labels', true).parent('#gui');
	checkLabels.changed(function(){
		showLabels = this.checked();
	});
	var checkPrey = createCheckbox('prey/predator mode', true).parent('#gui');
	checkPrey.changed(function(){
		preyWiring = this.checked();
		makeConnections();
	});
}

function mouseDragged() {
	if (dist(mouseX, mouseY, light.x, light.y) < 50) {
		light.x = mouseX;
		light.y = mouseY;
	}
}

function Light(x, y, intensity) {
	this.x = x;
	this.y = y;
	this.intensity = intensity;
	this.display = function() {
		push();
		noStroke();
		for (var d = 50; d > 1; d -= 2) {
			fill(255, 255, 0, 255 / d);
			ellipse(this.x, this.y, d, d);
		}
		pop();
	};
}
