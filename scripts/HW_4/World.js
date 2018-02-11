class World {
  constructor() {
    this.bkgdImage = createImage(width, height);
    
    // private
    this._tmin = 999;
    this._tmax = -999;
    this._scale = 0.015; // Perlin noise scale
    noiseSeed(9);

    // make a Perlin-noise background image
    for (let iy = 0; iy < height; iy++) {
      for (let ix = 0; ix < width; ix++) {
        let temp = noise(this._scale * ix, this._scale * iy);
        if (temp < this._tmin) this._tmin = temp
        if (temp > this._tmax) this._tmax = temp
      }
    }
    // renormalize
    for (let iy = 0; iy < height; iy++) {
      for (let ix = 0; ix < width; ix++) {
        let temp = noise(this._scale * ix, this._scale * iy) - this._tmin;
        temp /= (this._tmax - this._tmin);
        let redVal = Math.floor(255 * temp);
        let blueVal = Math.floor(255 * (1 - temp));
        let greenVal = min(redVal, blueVal);
        this.bkgdImage.set(ix, iy, color(redVal, greenVal, blueVal));
      }
    }
    this.bkgdImage.updatePixels();
  }

  display() {
    // draw the background image
    background(this.bkgdImage);
  }

  getTemperature(x, y) {
    // return the temperature at location (x, y)
    let temp = noise(this._scale * x, this._scale * y) - this._tmin;
    temp /= (this._tmax - this._tmin);
    return temp
  }
}