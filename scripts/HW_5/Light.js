function Light(x=random(width), y=random(height), intensity=1) {
	this.x = x;
	this.y = y;
	this.intensity = intensity;
}

Light.prototype.display = function () {
	// don't draw unless intensity > 0
	if (this.intensity <= 0) return;
	
	// draw a diffuse yellow light useing alpha shading
	push();
	translate(this.x, this.y);
	noStroke();
	for (var dia = 50; dia > 2; dia -= 2) {
		fill(255, 255, 100, 255/dia);
		ellipse(0, 0, dia, dia);
	}
	pop();
};