// Q learning with neural nets
// M. Nelson, Apr 2017

var itick = 0;
var bot;
var pellets = [];
var brain;

var simModes = ['training', 'testing', 'randAction'];
var modeSelector; // GUI element for selecting a controller
 
var faster = 1;
var paused = true;
var expt = null;

var NSTEPS = 2000;
var NTRIALS = 50;

var EXPT_SPEEDUP = 500;

//---------
// SETUP
//---------

function reset() {
	// reset bot
	bot.x = width / 2;
	bot.y = height - 10;
	bot.setController(modeSelector.value());
	bot.energy = 0;

	// reset pellets
	pellets = [];
	pellets.push(new Pellet());
	pellets.push(new Pellet());

	// reset sim
	itick = 0;
	paused = true;
}

function resetBrain() {
	var num_inputs = 4; //  [leftRed, leftGreen, rightRed, rightGreen]
	var num_actions = 3; // Left/Right/Stop
	var temporal_window = 0; // amount of temporal memory. 0 = agent lives in-the-moment :)
	var network_size = temporal_window * (num_inputs + num_actions) + num_inputs;

  // Specify Neural Network Architecture
	var layer_defs = [];
	layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
  // ADD HIDDEN LAYER(S) HERE
  layer_defs.push({type:'fc', num_neurons: 10, activation:'sigmoid'});
  layer_defs.push({type:'fc', num_neurons: 10, activation:'sigmoid'});
  layer_defs.push({type:'fc', num_neurons: 10, activation:'sigmoid'});
	layer_defs.push({type:'regression', num_neurons: num_actions});

	// options for the learning algorithm (feel free to edit these)
	var opt = {};
	opt.temporal_window = temporal_window;
	opt.experience_size = 10000;
	opt.start_learn_threshold = 2000;
	opt.gamma = 0.93;
	opt.learning_steps_total = 60000;
	opt.learning_steps_burnin = 10000;
	opt.epsilon_min = 0.05;
	opt.epsilon_test_time = 0.0;
	opt.layer_defs = layer_defs;
	opt.tdtrainer_options = {learning_rate:0.6, momentum:0, batch_size: 20, l2_decay:0};

	brain = new deepqlearn.Brain(num_inputs, num_actions, opt);
}

function setup() {
	createCanvas(390, 250).parent('canvas');
	createGUI();
	bot = new Bot();
	resetBrain();
	reset();
}

//------------
// MAIN LOOP
//------------

function draw() {
	background(0);
	for (var i=0; i<pellets.length; i++) {pellets[i].display();}
	bot.display();
	displayInfo();

	if (expt) {
		expt.update();
		expt.display();
		if (expt.isFinished()) {
			bot.setController(modeSelector.value());
			reset();
			expt = null;
		}
	} else {
		if (paused || itick >= NSTEPS) {
			select('#status').html('paused');
			noLoop();
		} else {
			var n = faster;
			while (n--) {
				simStep();
				if (itick >= NSTEPS) break;
			}
		}
	}
}

function simStep() {
	itick++;
	for (var i=0; i<pellets.length; i++) {pellets[i].update();}
	bot.update();
}

//---------
// DISPLAY
//---------

function displayInfo() {
	push();
	textSize(14);
	noStroke();
	fill('white');
	textAlign(LEFT);
	text(bot.controllerName, 5, 15);
	textAlign(CENTER);
	text("energy = " + bot.energy, width / 2, 15);
	textAlign(RIGHT);
	text(itick, width - 5, 15);
	pop();
}
//-------
// GUI
//-------
function createGUI() {

	// RESET BRAIN
	var qBtn = createButton('reset Brain').parent('#gui');
	qBtn.mousePressed(function() {
		resetBrain();
	});

	// CONTROLLER SELECTOR
	modeSelector = createSelect().parent('#gui');
	for (var i = 0; i < simModes.length; i++) {
		modeSelector.option(simModes[i]);
	}
	modeSelector.changed(function() {
		var val = this.value();
		bot.setController(val);
		if(val === "training") {brain.learning = true;}
		if(val === "testing")  {brain.learning = false;}
		reset();
		redraw();
	});

	// RESET
	var resetBtn = createButton('reset').parent('#gui');
	resetBtn.mousePressed(function() {
		reset();
		redraw();
	});

	// SINGLE STEP
	var stepBtn = createButton('single step').parent('#gui');
	stepBtn.mousePressed(function() {
		simStep();
		redraw();
	});

	// RUN/PAUSE
	var runBtn = createButton('run/pause').parent('#gui');
	runBtn.mousePressed(togglePause);


	// RUN SERIES
	var exptBtn = createButton('run series').parent('#gui');
	exptBtn.mousePressed(function() {
		expt = new Expt(EXPT_SPEEDUP);
		loop();
	});
	
	//-------
	// GUI2
	//-------

	// Save network
	var saveBtn = createButton('save network').parent('#gui2');
	saveBtn.mousePressed(function() {
	  var j = brain.value_net.toJSON();
	  var t = JSON.stringify(j);
	  select('#tt').html(t);
	});

	// Load network
	var loadBtn = createButton('load network').parent('#gui2');
	loadBtn.mousePressed(function() {
	  var t = select('#tt').html();
	  var j = JSON.parse(t);
	  brain.value_net.fromJSON(j);
	});

}

function togglePause() {
	paused = !paused;
	if (!paused) {
		select('#status').html('running');
		loop();
	}
}

//-------------
// EXPERIMENT
//------------

function Expt(speedup) {

	// run a set of trials for the currently selected controller
	// and enter stats in the data table

	this.done = false;
	this.itrial = 0;
	this.speedup = speedup;
	this.fitDat = [];

	reset();

	this.update = function() {

		// take n simulation steps
		var n = this.speedup;
		while (n--) {
			simStep();
			if (itick >= NSTEPS) break;
		}

		// update display
		this.display();

		if (itick >= NSTEPS) {
			// finished trial
			this.fitDat.push(bot.energy);
			reset();
			this.itrial++;
			if (this.itrial == NTRIALS) {
				// finished series
				var stats = calcArrayStats(this.fitDat);
				var buf = select('#table').html(); // get existing table data
				buf += "<tr>"; // add a new row
				buf += "<td>" + bot.controllerName + "</td>";
				buf += "<td>" + nf(stats.mean, 1, 2) + " (" + nf(stats.std, 1, 2) + ")" + "</td>";
				buf += "</tr>";
				select('#table').html(buf); // put it back in the table
				this.done = true;
				reset();
			}
		}

	};

	this.display = function() {
		select('#status').html(bot.controllerName + " trial/tick: " + [this.itrial, itick]);
	};

	this.isFinished = function() {
		return this.done;
	};
}

//------------
// UTILITIES
//------------

function calcArrayStats(inputArray) {
	/**
	 ** calculates mean, standard deviation and standar error 
	 **
	 ** input: inputArray, an array of numbers
	 ** returns: {mean: <mean>, std: <standard deviation>, sem: <standard error>}
	 */
	var sum = 0;
	var sumSq = 0;
	var n = inputArray.length;
	for (var i = 0; i < n; i++) {
		sum += inputArray[i];
		sumSq += inputArray[i] * inputArray[i];
	}
	var variance = (sumSq - (sum * sum) / n) / (n - 1);
	return {
		mean: sum / n,
		std: Math.sqrt(variance),
		sem: Math.sqrt(variance / n)
	};
}

function randint(min,max) {
	// return a random integer N such that min <= N <= max
    return Math.floor(Math.random()*(max-min+1)+min);
}

//------
// Bot
//------

// Bot constructor
function Bot(parms) {
  parms = parms || {};
  this.halfWidth = parms.halfWidth || 40;
  this.y = parms.y || height - 3;
  this.x = parms.x || random(this.halfWidth, width-this.halfWidth);
  this.vx = 0; // x velocity
  this.cfill = 'darkOrange';
  this.energy = 0;

  this.controllerName = '';
  if (typeof parms.controller === 'string') {
    this.setController(parms.controller);
  }

  this.sns = {
    left: [0, 0], // RED, GREEN
    right: [0, 0]
  };

  // action values
  this.LEFT = 0;
  this.RIGHT = 1;
  this.STOP = 2;

  // reinforcement learning
  this.state = [0, 0];
  this.action = 0;
  this.reward = 0;
  this.nextState = [0, 0];
}

Bot.prototype.consume = function() {
  // pellet consumption
  for (var i = 0; i < pellets.length; i++) {
    var dcheck = this.halfWidth + pellets[i].dia / 2;
      if (pellets[i].y > this.y && pellets[i].x >= this.x - dcheck && pellets[i].x < this.x + dcheck){
      this.reward += pellets[i].value;
      pellets[i].reset();
    }
  }
  this.energy += this.reward;
};

Bot.prototype.update = function() {
  this.reward = 0;
  this.state = this.getSensorState();
  this.controller();
  this.vx *= 0.98;
  this.vx = constrain(this.vx, -5, 5);
  this.x += this.vx;
  if (this.x < this.halfWidth) {
    this.x = this.halfWidth;
    this.vx = 0;
  } else if (this.x > width - this.halfWidth) {
    this.x = width - this.halfWidth;
    this.vx = 0;
  }
  
  this.consume();
  this.updateSensors();
  this.nextState = this.getSensorState();
  if (this.controllerName == "training") { 
    brain.backward(this.reward); 
  }
};

Bot.prototype.updateSensors = function() {
  this.sns.left = [0, 0];
  this.sns.right = [0, 0];
  for (var i=0; i < pellets.length; i++) {
    var idx = (pellets[i].color == 'red') ? 0 : 1;
    this.sns.left[idx] += 30.0 / dist(this.x - this.halfWidth, this.y, pellets[i].x, pellets[i].y);
    this.sns.right[idx] += 30.0 / dist(this.x + this.halfWidth, this.y, pellets[i].x, pellets[i].y);
  }
};

Bot.prototype.getSensorState = function() {
  return this.sns.left.concat(this.sns.right);
};

Bot.prototype.display = function() {
  push();
  fill(this.cfill); 
  translate(this.x, this.y);
  rect(-this.halfWidth, -3, 2*this.halfWidth, 6);
  /*
  textAlign(CENTER);
  text(nf(this.sns.left, 1, 2), -this.halfWidth, -6);
  text(nf(this.sns.right, 1, 2), this.halfWidth, -6);
  */
  pop();
};

Bot.prototype.setController = function(name) {
  this.controllerName = name;
  this.controller = this[name];
};

//-------------
// CONTROLLERS 
//-------------

Bot.prototype.randAction = function() {
  // pick a random action
  this.action = randint(0, 2);
  this.actionToMotor();
};

Bot.prototype.training = function() {
  this.action = brain.forward(this.state);
  this.actionToMotor();
};

Bot.prototype.testing = function() {
  this.action = brain.forward(this.state);
  this.actionToMotor();
};

Bot.prototype.actionToMotor = function() {
  if (this.action == this.LEFT) {
    this.vx -= 1;
  } else if (this.action == this.RIGHT) {
    this.vx += 1;
  } else if (this.action == this.STOP) {
    this.vx = 0;
  }
};


//---------
// Pellet
//---------

// Pellet constructor

function Pellet() {
    this.reset();
}

Pellet.prototype.display = function() {
	noStroke();
	fill(this.color);
	ellipse(this.x, this.y, this.dia, this.dia);
};

Pellet.prototype.reset = function() {
  
  this.dia = 20;
  this.color = (random() < 0.3) ? 'red' : 'green';  // red = bad (-1); green = good (+1)
  this.value = (this.color == 'red') ? -1 : 1;
  
  var xbuffer = (this.color == 'red') ? 100 : 30;  // provide safe zones from red pellets
  this.x = random(xbuffer, width - xbuffer);
  this.y = 50;
  this.vy = random(3,7);
};

Pellet.prototype.update = function() {
	this.y += this.vy;
	if (this.y > height) this.reset();
};