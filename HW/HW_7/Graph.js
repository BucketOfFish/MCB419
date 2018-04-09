//   Graph.js: multi-line graphs

// IN HTML:
//    <canvas id="fig1" width="400" height="300"></canvas>
// IN JS:
//   let legend = ["random", "sine"];
//   let graph = createGraph(legend, options);
//   for (let i = 0; i < 100; i++) {
//     graph.add(i, [Math.random(), 2*Math.sin(i/20)]);
//   }
//   graph.draw("fig1");

// USER FUNCTIONS:
//   graph.add(i, [y1 y2 ...]);  add y1 to line1, y2 to line2
//   graph.draw("canvas_id")     display graph in <canvas> element
//   graph.reset()               clear graph data
//   graph.get()                 return data arrays and data limits

// options: {
//   stepHorizon: STEPS,          scalar, initial x range (expands automatically)
//   type: ['line', 'scatter']    array, same length as legend
// }


function createGraph(legend = "no legend", options = {}) {
  var pts = [];
  var nLines = 0;
  var xdat = {
    min: Infinity,
    max: -Infinity
  };
  var ydat = {
    min: Infinity,
    max: -Infinity
  };

  var stepHorizon = false; // true for continuously expanding x range

  var styles = ["red", "green", "blue", "cyan", "magenta", "black", "purple", "aqua", "olive", "lime", "navy"];

  if (Array.isArray(legend)) {
    nLines = legend.length;
  } else {
    // user provided a string, make it a singleton array
    nLines = 1;
    legend = [legend];
  }

  if (options.stepHorizon) {
    stepHorizon = {
      current: options.stepHorizon,
      original: options.stepHorizon
    };
  }

  //========
  // add
  //========

  function add(xpt, y) {

    if (xpt > xdat.max) xdat.max = xpt;
    if (xpt < xdat.min) xdat.min = xpt;

    if (stepHorizon && xdat.max > stepHorizon.current) stepHorizon.current *= 2;

    if (!Array.isArray(y)) y = [y];

    if (y.length !== nLines) {
      console.log('graph error: legend/data length mismatch');
      return;
    }
    for (let i = 0; i < nLines; i++) {
      let ypt = y[i];
      if (ypt > ydat.max) ydat.max = ypt;
      if (ypt < ydat.min) ydat.min = ypt;
    }

    pts.push({
      xpt: xpt,
      y: y
    });

  }

  //===========
  // calcTicks
  //===========

  function calcTicks(vmin, vmax, ntry) {
    // ntry is number of ticks to try
    let tickStep = (vmax - vmin) / (ntry - 1);
    let mag = Math.pow(10, Math.floor(Math.log10(tickStep)));
    let residual = tickStep / mag;
    if (residual > 5) {
      tickStep = 10 * mag;
    } else if (residual > 2) {
      tickStep = 5 * mag;
    } else if (residual > 1) {
      tickStep = 2 * mag;
    } else {
      tickStep = mag;
    }
    let tickMin = tickStep * Math.floor(vmin / tickStep);
    let tickMax = tickStep * Math.ceil(vmax / tickStep);
    let nticks = Math.floor((tickMax - tickMin) / tickStep) + 1;
    return {
      min: tickMin,
      max: tickMax,
      step: tickStep,
      n: nticks
    };
  }

  //========
  // draw
  //========

  function draw(canvID) {
    
    // remove leading #, if present
    if(canvID[0] == "#")canvID = canvID.slice(1);

    let canv = document.getElementById(canvID);
    let ctx = canv.getContext('2d');
    ctx.font = "10px Georgia";

    let H = canv.height;
    let W = canv.width;
    let pad = 25;

    let xticks = {
      min: 0,
      max: 1,
      step: 0.2,
      n: 6
    };
    let yticks = {
      min: 0,
      max: 1,
      step: 0.2,
      n: 6
    };

    function f2t(x) {
      return '' + Math.round(x * 100) / 100;
    }

    // transform from (x,y) to canvas (tx,ty)
    function t(x, y) {
      let tx = ((x - xticks.min) / (xticks.max - xticks.min)) * (W - pad * 2) + pad;
      let ty = H - ((y - yticks.min) / (yticks.max - yticks.min) * (H - pad * 2) + pad);
      return {
        tx,
        ty
      };
    }

    ctx.clearRect(0, 0, W, H);

    let N = pts.length;
    if (N > 1) {
      let xmin = stepHorizon ? 0 : xdat.min;
      let xmax = stepHorizon ? stepHorizon.current : xdat.max;
      xticks = calcTicks(xmin, xmax, 10);
      yticks = calcTicks(ydat.min, ydat.max, 10);
    }

    // draw guidelines and values
    ctx.fillStyle = "#999";
    ctx.strokeStyle = "#999";
    ctx.beginPath();
    // xticks
    for (let i = 0; i <= xticks.n; i++) {
      let xpos = i / (xticks.n - 1) * (W - 2 * pad) + pad;
      ctx.moveTo(xpos, pad);
      ctx.lineTo(xpos, H - pad);
      ctx.fillText(f2t(xticks.step * i + xticks.min), xpos, H - pad + 14);
    }
    // yticks
    for (let i = 0; i < yticks.n; i++) {
      let ypos = i / (yticks.n - 1) * (H - 2 * pad) + pad;
      ctx.moveTo(pad, ypos);
      ctx.lineTo(W - pad, ypos);
      ctx.fillText(f2t((yticks.n - 1 - i) * yticks.step + yticks.min), 0, ypos);
    }
    ctx.stroke();

    if (N < 2) return;

    // draw data
    for (let i = 0; i < nLines; i++) {
      if (options.type && options.type[i] == 'scatter') {
        // scatter
        ctx.strokeStyle = styles[i % styles.length];
        ctx.beginPath();
        for (let j = 0; j < N; j++) {
          // draw line from j-1 to j
          let p = pts[j];
          let pt = t(p.xpt, p.y[i]);
          ctx.rect(pt.tx - 2, pt.ty - 2, 4, 4);
        }
        ctx.stroke();

      } else {
        // line graph
        ctx.strokeStyle = styles[i % styles.length];
        ctx.beginPath();
        for (let j = 0; j < N; j++) {
          // draw line from j-1 to j
          p = pts[j];
          pt = t(p.xpt, p.y[i]);
          if (j === 0) ctx.moveTo(pt.tx, pt.ty);
          else ctx.lineTo(pt.tx, pt.ty);
        }
        ctx.stroke();
      }
    }

    // draw legend
    let legendX = 100; // pixels from left edge
    let legendY = 5; // pixels from top edge
    let legendW = 30; // min width
    let legendH = 16 * nLines;
    let legendPad = 5;

    // calc legend width
    for (let i = 0; i < legend.length; i++) {
      var textW = ctx.measureText(legend[i]).width;
      if (textW > legendW) legendW = textW;
    }

    // legend background - grey box with padding
    ctx.fillStyle = "#eee";
    ctx.fillRect(
      (W - pad) - legendX - legendPad,
      pad + legendY,
      legendW + 2 * legendPad,
      legendH + 2 * legendPad);

    // legend text
    for (let i = 0; i < nLines; i++) {
      ctx.fillStyle = styles[i % styles.length];
      ctx.fillText(legend[i], W - pad - 100, pad + 20 + i * 16);
    }

  }

  //========
  // get
  //========

  function get() {
    return {
      xdat,
      ydat,
      pts
    };
  }

  //========
  // resest
  //========

  function reset() {
    pts = [];
    xdat = {
      min: Infinity,
      max: -Infinity
    };
    ydat = {
      min: Infinity,
      max: -Infinity
    };
    if (stepHorizon) stepHorizon.current = stepHorizon.original;
  }
  
  //=======================
  // return user functions
  //=======================

  return {
    add,
    draw,
    get,
    reset
  }
}