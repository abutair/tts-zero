const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'VITS2 From-Scratch Research';
pptx.subject = 'Learning progress through tensor foundations, digital audio, and STFT preparation';
pptx.title = 'VITS2 From Scratch: Foundations to STFT';
pptx.company = 'Learning apprenticeship';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Aptos Display',
  bodyFontFace: 'Aptos',
  lang: 'en-US',
};
pptx.defineSlideMaster({
  title: 'MASTER',
  background: { color: 'F6F8FC' },
  objects: [
    { line: { x: 0.55, y: 7.18, w: 12.25, h: 0, line: { color: 'D7DEEA', width: 1 } } },
    { text: { text: 'VITS2 from scratch - learning progress', options: { x: 0.62, y: 7.22, w: 5.2, h: 0.18, fontFace: 'Aptos', fontSize: 8.5, color: '667085', margin: 0 } } },
  ],
  slideNumber: { x: 12.25, y: 7.2, w: 0.45, h: 0.2, fontFace: 'Aptos', fontSize: 9, color: '667085', align: 'right', margin: 0 },
});

const C = {
  navy: '14213D',
  blue: '2563EB',
  blue2: '6EA8FE',
  coral: 'D94F70',
  green: '168A6B',
  amber: 'C98716',
  text: '172033',
  muted: '667085',
  grid: 'D7DEEA',
  paleBlue: 'EAF1FF',
  paleCoral: 'FCECF0',
  paleGreen: 'E8F7F1',
  paleAmber: 'FFF4DB',
  white: 'FFFFFF',
};

const ST = pptx.ShapeType;

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.65, y: 0.42, w: 12.0, h: 0.48,
    fontFace: 'Aptos Display', fontSize: 25, bold: true, color: C.navy,
    margin: 0, fit: 'shrink',
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.67, y: 0.94, w: 11.9, h: 0.36,
      fontFace: 'Aptos', fontSize: 12.5, color: C.muted,
      margin: 0, fit: 'shrink',
    });
  }
}

function addPill(slide, text, x, y, w, fill, color = C.text) {
  slide.addShape(ST.roundRect, {
    x, y, w, h: 0.38,
    rectRadius: 0.08,
    fill: { color: fill }, line: { color: fill },
  });
  slide.addText(text, {
    x: x + 0.08, y: y + 0.06, w: w - 0.16, h: 0.22,
    fontSize: 10.5, bold: true, align: 'center', color, margin: 0, fit: 'shrink',
  });
}

function addCard(slide, x, y, w, h, title, body, fill = C.white, accent = C.blue) {
  slide.addShape(ST.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    fill: { color: fill }, line: { color: 'D9E0EC', width: 1 },
    shadow: { type: 'outer', color: 'B8C2D3', opacity: 0.16, blur: 1, angle: 45, distance: 1 },
  });
  slide.addShape(ST.rect, { x, y, w: 0.08, h, fill: { color: accent }, line: { color: accent } });
  slide.addText(title, {
    x: x + 0.22, y: y + 0.18, w: w - 0.4, h: 0.3,
    fontSize: 15, bold: true, color: C.text, margin: 0, fit: 'shrink',
  });
  slide.addText(body, {
    x: x + 0.22, y: y + 0.58, w: w - 0.42, h: h - 0.74,
    fontSize: 11.5, color: C.text, margin: 0.02, breakLine: false,
    valign: 'top', fit: 'shrink',
  });
}

function addArrow(slide, x1, y1, x2, y2, color = C.blue) {
  slide.addShape(ST.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width: 2.2, endArrowType: 'triangle' },
  });
}

function addMatrix(slide, x, y, rows, cols, values, validFn, cellW = 0.48, cellH = 0.42) {
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const valid = validFn(r, c);
      slide.addShape(ST.rect, {
        x: x + c * cellW, y: y + r * cellH, w: cellW - 0.03, h: cellH - 0.03,
        fill: { color: valid ? C.paleBlue : 'EDF0F5' },
        line: { color: valid ? C.blue2 : 'CDD5E1', width: 0.8 },
      });
      slide.addText(String(values[r][c]), {
        x: x + c * cellW, y: y + r * cellH + 0.09, w: cellW - 0.03, h: 0.2,
        fontSize: 10, color: valid ? C.text : C.muted, align: 'center', margin: 0,
      });
    }
  }
}

function addWave(slide, x, y, w, h, cycles = 2, color = C.blue, width = 2.2) {
  const points = [];
  const n = 80;
  for (let i = 0; i <= n; i += 1) {
    const px = x + (i / n) * w;
    const py = y + h / 2 - Math.sin((i / n) * Math.PI * 2 * cycles) * h * 0.4;
    points.push([px, py]);
  }
  for (let i = 0; i < points.length - 1; i += 1) {
    slide.addShape(ST.line, {
      x: points[i][0], y: points[i][1],
      w: points[i + 1][0] - points[i][0], h: points[i + 1][1] - points[i][1],
      line: { color, width, beginArrowType: 'none', endArrowType: 'none' },
    });
  }
}

// 1. Title
{
  const slide = pptx.addSlide('MASTER');
  slide.background = { color: C.navy };
  slide.addShape(ST.ellipse, { x: 8.9, y: -1.2, w: 5.3, h: 5.3, fill: { color: '244B91', transparency: 28 }, line: { color: '244B91', transparency: 100 } });
  slide.addShape(ST.ellipse, { x: 10.1, y: 3.9, w: 3.8, h: 3.8, fill: { color: C.coral, transparency: 18 }, line: { color: C.coral, transparency: 100 } });
  slide.addText('VITS2 from scratch', {
    x: 0.9, y: 1.25, w: 7.7, h: 0.8,
    fontFace: 'Aptos Display', fontSize: 40, bold: true, color: C.white, margin: 0,
  });
  slide.addText('Foundations to STFT', {
    x: 0.92, y: 2.15, w: 7.3, h: 0.62,
    fontFace: 'Aptos Display', fontSize: 29, color: C.blue2, margin: 0,
  });
  slide.addText('A visual recap of what we have learned and verified so far', {
    x: 0.95, y: 3.05, w: 6.8, h: 0.55,
    fontSize: 17, color: 'DCE6FA', margin: 0,
  });
  addPill(slide, '01 Tensor foundations', 0.95, 4.25, 2.25, 'EAF1FF', C.navy);
  addPill(slide, '02 Digital audio', 3.38, 4.25, 2.0, 'E8F7F1', C.navy);
  addPill(slide, '03 STFT checkpoint', 5.56, 4.25, 2.1, 'FCECF0', C.navy);
  slide.addText('Current verified signal: 1 second, 440 Hz, float32, shape [1, 22050]', {
    x: 0.95, y: 5.2, w: 7.9, h: 0.45, fontSize: 14, color: C.white, margin: 0,
  });
}

// 2. Learning path
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'The dependency-first learning path', 'Each chapter establishes invariants required by the next one.');
  const xs = [0.8, 3.55, 6.3, 9.05];
  const titles = ['Tensor contracts', 'Digital waveform', 'Time-frequency view', 'VITS2 acoustic path'];
  const bodies = [
    '[B, T], [B, C, T]\nLengths, padding, masks\nBroadcasting',
    'Sample index -> time\nFrequency and cycles\nSine amplitude',
    'Windows and hops\nComplex STFT\nMagnitude and mel',
    'Posterior encoder\nAlignment and flow\nDecoder and losses',
  ];
  const fills = [C.paleBlue, C.paleGreen, C.paleCoral, C.paleAmber];
  const accents = [C.blue, C.green, C.coral, C.amber];
  for (let i = 0; i < 4; i += 1) {
    addCard(slide, xs[i], 2.05, 2.3, 2.35, titles[i], bodies[i], fills[i], accents[i]);
    if (i < 3) addArrow(slide, xs[i] + 2.35, 3.2, xs[i + 1] - 0.1, 3.2, C.muted);
  }
  slide.addText('Current position', { x: 6.74, y: 4.65, w: 1.4, h: 0.3, fontSize: 11, bold: true, color: C.coral, align: 'center', margin: 0 });
  slide.addShape(ST.chevron, { x: 7.2, y: 4.98, w: 0.48, h: 0.45, rotate: 270, fill: { color: C.coral }, line: { color: C.coral } });
  slide.addText('Why this order matters', { x: 0.9, y: 5.38, w: 2.5, h: 0.34, fontSize: 16, bold: true, color: C.navy, margin: 0 });
  slide.addText('A model can have correct layer code and still fail because its shapes, masks, timestamps, or acoustic features are semantically wrong.', {
    x: 0.9, y: 5.86, w: 11.45, h: 0.62, fontSize: 16, color: C.text, margin: 0.02, fit: 'shrink',
  });
}

// 3. Tensor foundations
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Chapter 01 - tensor shapes are contracts', 'The axes describe meaning, not just storage.');
  addCard(slide, 0.75, 1.55, 3.55, 1.45, '[B, T]', 'Raw token IDs or waveform samples\nB = batch, T = sequence time', C.paleBlue, C.blue);
  addCard(slide, 4.88, 1.55, 3.55, 1.45, '[B, C, T]', 'Encoded features\nC = feature channels', C.paleGreen, C.green);
  addCard(slide, 9.0, 1.55, 3.55, 1.45, '[B, 1, T]', 'Sequence mask\n1 broadcasts across channels', C.paleCoral, C.coral);

  slide.addText('Two padded text sequences', { x: 0.85, y: 3.5, w: 3.2, h: 0.32, fontSize: 15, bold: true, color: C.navy, margin: 0 });
  const values = [[11, 12, 13, 0, 0], [21, 22, 23, 24, 25]];
  addMatrix(slide, 0.9, 4.05, 2, 5, values, (r, c) => c < [3, 5][r], 0.58, 0.52);
  slide.addText('lengths = [3, 5]', { x: 0.9, y: 5.25, w: 2.4, h: 0.3, fontSize: 12.5, color: C.muted, margin: 0 });

  slide.addText('Mask meaning', { x: 4.4, y: 3.5, w: 2.3, h: 0.32, fontSize: 15, bold: true, color: C.navy, margin: 0 });
  slide.addText('position < length', { x: 4.4, y: 4.1, w: 2.55, h: 0.45, fontFace: 'Consolas', fontSize: 18, bold: true, color: C.blue, margin: 0 });
  slide.addText('True = real data\nFalse = padding', { x: 4.4, y: 4.85, w: 2.4, h: 0.8, fontSize: 14, color: C.text, margin: 0 });

  slide.addText('Broadcasting result', { x: 7.5, y: 3.5, w: 2.6, h: 0.32, fontSize: 15, bold: true, color: C.navy, margin: 0 });
  slide.addText('[B, C, T]\n× [B, 1, T]\n= [B, C, T]', {
    x: 7.55, y: 4.0, w: 2.5, h: 1.45, fontFace: 'Consolas', fontSize: 18, bold: true, color: C.green, align: 'center', margin: 0,
  });
  addCard(slide, 10.15, 3.62, 2.4, 2.1, 'Key warning', '[B, T] × [B, 1, T]\ncan accidentally produce\n[B, B, T].', C.paleAmber, C.amber);
}

// 4. Mask construction
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'From lengths to a Boolean mask', 'Broadcasting compares every sample length with every possible position.');
  addCard(slide, 0.8, 1.55, 3.1, 1.55, 'Lengths [B, 1]', '[[3],\n [5]]', C.paleBlue, C.blue);
  addCard(slide, 5.05, 1.55, 3.1, 1.55, 'Positions [1, T]', '[[0, 1, 2, 3, 4]]', C.paleGreen, C.green);
  addArrow(slide, 3.98, 2.3, 4.88, 2.3, C.muted);
  slide.addText('position < length', { x: 4.2, y: 1.65, w: 1.45, h: 0.3, fontSize: 10, color: C.muted, align: 'center', margin: 0 });
  addArrow(slide, 8.22, 2.3, 9.08, 2.3, C.muted);
  addCard(slide, 9.15, 1.55, 3.35, 1.55, 'Mask [B, T]', 'T T T F F\nT T T T T', C.paleCoral, C.coral);

  slide.addText('Why unsqueeze matters', { x: 0.85, y: 3.75, w: 3.1, h: 0.38, fontSize: 17, bold: true, color: C.navy, margin: 0 });
  const labels = [
    ['[T]', 'positions'], ['[1, T]', 'unsqueeze(0)'], ['[B]', 'lengths'], ['[B, 1]', 'unsqueeze(1)'], ['[B, T]', 'comparison'], ['[B, 1, T]', 'channel mask'],
  ];
  const xx = [0.9, 2.85, 4.8, 6.75, 8.7, 10.65];
  labels.forEach((item, i) => {
    slide.addShape(ST.roundRect, { x: xx[i], y: 4.45, w: 1.55, h: 0.85, fill: { color: i % 2 ? C.paleGreen : C.paleBlue }, line: { color: C.grid }, rectRadius: 0.05 });
    slide.addText(item[0], { x: xx[i], y: 4.59, w: 1.55, h: 0.27, fontFace: 'Consolas', fontSize: 15, bold: true, color: C.text, align: 'center', margin: 0 });
    slide.addText(item[1], { x: xx[i], y: 4.93, w: 1.55, h: 0.2, fontSize: 9.5, color: C.muted, align: 'center', margin: 0 });
    if (i < labels.length - 1) addArrow(slide, xx[i] + 1.58, 4.88, xx[i + 1] - 0.08, 4.88, C.muted);
  });
  slide.addText('Invariant: mask.sum() == lengths.sum()', { x: 3.95, y: 5.88, w: 5.25, h: 0.42, fontFace: 'Consolas', fontSize: 16, bold: true, color: C.green, align: 'center', margin: 0 });
}

// 5. Digital audio
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Chapter 02 - a waveform is a timed sequence of measurements', 'Index, time, and amplitude answer three different questions.');
  slide.addText('Sample rate = 4 measurements per second', { x: 0.85, y: 1.4, w: 4.4, h: 0.38, fontSize: 17, bold: true, color: C.navy, margin: 0 });
  slide.addShape(ST.line, { x: 1.15, y: 3.2, w: 10.65, h: 0, line: { color: C.grid, width: 2 } });
  const amps = [0, 0.8, 0.2, -0.7];
  const times = ['0.00 s', '0.25 s', '0.50 s', '0.75 s'];
  for (let i = 0; i < 4; i += 1) {
    const x = 1.3 + i * 3.35;
    const y = 3.2 - amps[i] * 1.1;
    slide.addShape(ST.line, { x, y: 3.2, w: 0, h: y - 3.2, line: { color: C.blue, width: 2 } });
    slide.addShape(ST.ellipse, { x: x - 0.1, y: y - 0.1, w: 0.2, h: 0.2, fill: { color: C.blue }, line: { color: C.blue } });
    slide.addText(`index ${i}`, { x: x - 0.55, y: 4.0, w: 1.1, h: 0.25, fontSize: 11, bold: true, color: C.text, align: 'center', margin: 0 });
    slide.addText(times[i], { x: x - 0.55, y: 4.32, w: 1.1, h: 0.25, fontSize: 10.5, color: C.muted, align: 'center', margin: 0 });
    slide.addText(`amp ${amps[i]}`, { x: x - 0.62, y: y - 0.48, w: 1.24, h: 0.22, fontSize: 10.5, color: C.blue, align: 'center', margin: 0 });
  }
  addCard(slide, 0.85, 5.25, 3.45, 1.05, 'Index', 'Which stored measurement?', C.paleBlue, C.blue);
  addCard(slide, 4.92, 5.25, 3.45, 1.05, 'Time', 'When was it measured?', C.paleGreen, C.green);
  addCard(slide, 8.98, 5.25, 3.45, 1.05, 'Amplitude', 'What value was measured?', C.paleCoral, C.coral);
}

// 6. Frequency and cycles
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Frequency controls repetition; amplitude controls height', 'Do not confuse pitch with loudness.');
  slide.addText('1 Hz: one cycle per second', { x: 0.8, y: 1.5, w: 3.6, h: 0.35, fontSize: 16, bold: true, color: C.navy, margin: 0 });
  addWave(slide, 0.85, 2.0, 5.3, 1.35, 1, C.blue, 2.6);
  slide.addShape(ST.line, { x: 0.85, y: 2.68, w: 5.3, h: 0, line: { color: C.grid, width: 1 } });
  slide.addText('0 s', { x: 0.75, y: 3.43, w: 0.55, h: 0.22, fontSize: 10, color: C.muted, margin: 0 });
  slide.addText('1 s', { x: 5.82, y: 3.43, w: 0.55, h: 0.22, fontSize: 10, color: C.muted, margin: 0 });

  slide.addText('3 Hz: three cycles per second', { x: 6.85, y: 1.5, w: 4.2, h: 0.35, fontSize: 16, bold: true, color: C.navy, margin: 0 });
  addWave(slide, 6.9, 2.0, 5.3, 1.35, 3, C.coral, 2.6);
  slide.addShape(ST.line, { x: 6.9, y: 2.68, w: 5.3, h: 0, line: { color: C.grid, width: 1 } });
  slide.addText('0 s', { x: 6.8, y: 3.43, w: 0.55, h: 0.22, fontSize: 10, color: C.muted, margin: 0 });
  slide.addText('1 s', { x: 11.87, y: 3.43, w: 0.55, h: 0.22, fontSize: 10, color: C.muted, margin: 0 });

  addCard(slide, 0.85, 4.35, 3.55, 1.45, 'Time from index', 'time = sample_index / sample_rate', C.paleBlue, C.blue);
  addCard(slide, 4.88, 4.35, 3.55, 1.45, 'Cycle progress', 'cycles = frequency × time', C.paleGreen, C.green);
  addCard(slide, 8.9, 4.35, 3.55, 1.45, 'Sine amplitude', 'sin(2π × cycles)', C.paleCoral, C.coral);
  slide.addText('The factor 2 matters: sin(π × cycles) would halve the intended frequency.', { x: 2.05, y: 6.15, w: 9.2, h: 0.4, fontSize: 14, bold: true, color: C.coral, align: 'center', margin: 0 });
}

// 7. 440 Hz verified waveform
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'The controlled 440 Hz waveform checkpoint', 'A known signal becomes ground truth for the STFT experiment.');
  const steps = [
    ['Indices', '[22050]'], ['Time', '[22050]'], ['Cycles', '[22050]'], ['Amplitude', '[22050]'], ['Batch', '[1, 22050]'],
  ];
  const x0 = 0.75;
  steps.forEach((s, i) => {
    const x = x0 + i * 2.5;
    slide.addShape(ST.roundRect, { x, y: 1.72, w: 1.85, h: 1.05, fill: { color: i === 4 ? C.paleCoral : C.paleBlue }, line: { color: i === 4 ? C.coral : C.blue2 }, rectRadius: 0.06 });
    slide.addText(s[0], { x, y: 1.94, w: 1.85, h: 0.26, fontSize: 14, bold: true, color: C.text, align: 'center', margin: 0 });
    slide.addText(s[1], { x, y: 2.3, w: 1.85, h: 0.22, fontFace: 'Consolas', fontSize: 11.5, color: C.muted, align: 'center', margin: 0 });
    if (i < steps.length - 1) addArrow(slide, x + 1.9, 2.23, x + 2.35, 2.23, C.muted);
  });
  addWave(slide, 0.95, 3.35, 11.35, 1.2, 8, C.blue, 2.2);
  slide.addShape(ST.line, { x: 0.95, y: 3.95, w: 11.35, h: 0, line: { color: C.grid, width: 1 } });

  const metrics = [
    ['Shape', '[1, 22050]', C.paleBlue, C.blue],
    ['Dtype', 'float32', C.paleGreen, C.green],
    ['Range', '≈ -1 to +1', C.paleAmber, C.amber],
    ['Upward crossings', '439 (440 ± 1)', C.paleCoral, C.coral],
  ];
  metrics.forEach((m, i) => addCard(slide, 0.8 + i * 3.12, 5.02, 2.7, 1.2, m[0], m[1], m[2], m[3]));
}

// 8. Why STFT
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Chapter 03 - why the “short-time” part matters', 'A whole FFT tells us what exists; an STFT also preserves approximately when.');
  addCard(slide, 0.8, 1.55, 5.6, 2.3, 'Recording A: sequential', 'First half: 440 Hz\nSecond half: 880 Hz', C.paleBlue, C.blue);
  addWave(slide, 1.25, 2.45, 2.1, 0.75, 3, C.blue, 2);
  addWave(slide, 3.65, 2.45, 2.1, 0.75, 6, C.coral, 2);
  slide.addShape(ST.line, { x: 3.5, y: 2.3, w: 0, h: 1.05, line: { color: C.grid, width: 1.5, dash: 'dash' } });

  addCard(slide, 6.95, 1.55, 5.6, 2.3, 'Recording B: simultaneous', '440 Hz and 880 Hz\nplay together for the whole second', C.paleCoral, C.coral);
  addWave(slide, 7.4, 2.42, 4.65, 0.72, 3, C.blue, 1.6);
  addWave(slide, 7.4, 2.42, 4.65, 0.72, 6, C.coral, 1.6);

  addCard(slide, 0.85, 4.55, 5.45, 1.35, 'Whole FFT', 'May show both 440 Hz and 880 Hz\nwithout clear timing.', C.paleAmber, C.amber);
  addCard(slide, 6.98, 4.55, 5.45, 1.35, 'STFT', 'Analyzes short frames and retains a\ntime sequence of frequency content.', C.paleGreen, C.green);
  addArrow(slide, 6.28, 5.22, 6.88, 5.22, C.muted);
}

// 9. Windows and hops
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'STFT framing: window, hop, and overlap', 'The current project uses 1024-sample windows starting every 256 samples.');
  addWave(slide, 0.8, 1.45, 11.7, 1.0, 12, C.blue, 1.7);
  slide.addShape(ST.line, { x: 0.8, y: 1.95, w: 11.7, h: 0, line: { color: C.grid, width: 1 } });
  const frameX = [0.8, 2.25, 3.7];
  for (let i = 0; i < 3; i += 1) {
    slide.addShape(ST.rect, { x: frameX[i], y: 2.65 + i * 0.65, w: 5.8, h: 0.44, fill: { color: C.blue, transparency: 85 }, line: { color: C.blue, width: 1.4 } });
    slide.addText(`Frame ${i}`, { x: frameX[i] + 0.12, y: 2.75 + i * 0.65, w: 1.0, h: 0.2, fontSize: 10.5, color: C.text, margin: 0 });
  }
  slide.addShape(ST.line, { x: 0.8, y: 4.74, w: 1.45, h: 0, line: { color: C.coral, width: 2.5, beginArrowType: 'triangle', endArrowType: 'triangle' } });
  slide.addText('hop = 256', { x: 1.05, y: 4.86, w: 1.1, h: 0.25, fontSize: 10.5, color: C.coral, align: 'center', margin: 0 });

  addCard(slide, 0.8, 5.45, 2.75, 1.05, 'Window duration', '46.44 ms', C.paleBlue, C.blue);
  addCard(slide, 3.78, 5.45, 2.75, 1.05, 'Hop duration', '11.61 ms', C.paleGreen, C.green);
  addCard(slide, 6.76, 5.45, 2.75, 1.05, 'Overlap', '768 samples = 75%', C.paleAmber, C.amber);
  addCard(slide, 9.74, 5.45, 2.75, 1.05, 'Expected frames', '87 with center=True', C.paleCoral, C.coral);
}

// 10. Complex STFT
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Complex STFT values: strength plus phase', 'Every frequency bin in every time frame is represented by a complex number.');
  slide.addShape(ST.ellipse, { x: 1.05, y: 1.65, w: 3.0, h: 3.0, fill: { color: C.white, transparency: 100 }, line: { color: C.grid, width: 1.5 } });
  slide.addShape(ST.line, { x: 1.35, y: 3.15, w: 2.4, h: 0, line: { color: C.grid, width: 1.2 } });
  slide.addShape(ST.line, { x: 2.55, y: 1.95, w: 0, h: 2.4, line: { color: C.grid, width: 1.2 } });
  slide.addShape(ST.line, { x: 2.55, y: 3.15, w: 1.0, h: -1.34, line: { color: C.coral, width: 3, endArrowType: 'triangle' } });
  slide.addText('real = 3', { x: 2.72, y: 3.25, w: 1.0, h: 0.25, fontSize: 11, color: C.blue, margin: 0 });
  slide.addText('imag = 4', { x: 3.63, y: 2.25, w: 1.0, h: 0.25, fontSize: 11, color: C.coral, margin: 0 });
  slide.addText('|3 + 4j| = 5', { x: 1.6, y: 4.86, w: 2.0, h: 0.35, fontFace: 'Consolas', fontSize: 17, bold: true, color: C.navy, align: 'center', margin: 0 });

  addCard(slide, 5.05, 1.55, 3.3, 1.5, 'Complex STFT', '[B, F, T_frames]\ncomplex64', C.paleCoral, C.coral);
  addArrow(slide, 8.48, 2.3, 9.05, 2.3, C.muted);
  addCard(slide, 9.18, 1.55, 3.3, 1.5, 'Magnitude', '[B, F, T_frames]\nfloat32, nonnegative', C.paleGreen, C.green);
  addCard(slide, 5.05, 3.8, 3.3, 1.5, 'Magnitude', 'How strong is this\nfrequency component?', C.paleBlue, C.blue);
  addCard(slide, 9.18, 3.8, 3.3, 1.5, 'Phase', 'Where is the oscillation\ninside its cycle?', C.paleAmber, C.amber);
  slide.addText('Taking magnitude changes the representation, not the tensor shape.', { x: 5.25, y: 5.87, w: 7.0, h: 0.4, fontSize: 14.5, bold: true, color: C.navy, align: 'center', margin: 0 });
}

// 11. Frequency bins
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'FFT bins are a discrete frequency ruler', 'With n_fft = 1024, neighboring bin centers are 21.53 Hz apart.');
  slide.addText('bin spacing = sample_rate / n_fft = 22050 / 1024 ≈ 21.53 Hz', { x: 0.85, y: 1.38, w: 6.4, h: 0.38, fontFace: 'Consolas', fontSize: 15, bold: true, color: C.blue, margin: 0 });
  const bins = [18, 19, 20, 21, 22, 23];
  const hz = [387.6, 409.1, 430.7, 452.2, 473.7, 495.3];
  const heights = [0.25, 0.75, 2.2, 1.45, 0.45, 0.18];
  const x0 = 1.0;
  slide.addShape(ST.line, { x: x0, y: 5.25, w: 11.2, h: 0, line: { color: C.grid, width: 1.5 } });
  for (let i = 0; i < bins.length; i += 1) {
    const x = x0 + i * 2.1;
    slide.addShape(ST.line, { x, y: 2.15, w: 0, h: 3.2, line: { color: C.grid, width: 1 } });
    slide.addShape(ST.rect, { x: x - 0.28, y: 5.25 - heights[i], w: 0.56, h: heights[i], fill: { color: C.blue }, line: { color: C.blue } });
    slide.addText(`bin ${bins[i]}`, { x: x - 0.48, y: 5.42, w: 0.96, h: 0.22, fontSize: 10.5, bold: true, color: C.text, align: 'center', margin: 0 });
    slide.addText(`${hz[i]} Hz`, { x: x - 0.58, y: 5.72, w: 1.16, h: 0.22, fontSize: 9.5, color: C.muted, align: 'center', margin: 0 });
  }
  const targetX = x0 + 2 * 2.1 + (440 - 430.7) / (452.2 - 430.7) * 2.1;
  slide.addShape(ST.line, { x: targetX, y: 1.95, w: 0, h: 3.4, line: { color: C.coral, width: 2.5, dash: 'dash' } });
  slide.addText('440 Hz target', { x: targetX - 0.75, y: 1.74, w: 1.5, h: 0.28, fontSize: 12, bold: true, color: C.coral, align: 'center', margin: 0 });
  slide.addText('fractional bin ≈ 20.43', { x: 8.65, y: 1.35, w: 3.0, h: 0.32, fontFace: 'Consolas', fontSize: 14, bold: true, color: C.navy, align: 'right', margin: 0 });
  slide.addText('The tone lies between bins 20 and 21. Bin 20 is closer, while neighboring bins retain energy because of spectral leakage.', {
    x: 1.35, y: 6.25, w: 10.7, h: 0.42, fontSize: 13.5, color: C.text, align: 'center', margin: 0, fit: 'shrink',
  });
}

// 12. What comes next
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Next checkpoint: waveform to model-ready log-mel features', 'The controlled 440 Hz signal will verify each transformation.');
  const nodes = [
    ['Waveform', '[1, 22050]', C.paleBlue, C.blue],
    ['Complex STFT', '[1, 513, 87]', C.paleCoral, C.coral],
    ['Magnitude', '[1, 513, 87]', C.paleGreen, C.green],
    ['Mel', '[1, 80, 87]', C.paleAmber, C.amber],
    ['Log-mel', '[1, 80, 87]', C.paleBlue, C.blue],
  ];
  nodes.forEach((n, i) => {
    const x = 0.55 + i * 2.58;
    slide.addShape(ST.roundRect, { x, y: 1.8, w: 2.0, h: 1.25, fill: { color: n[2] }, line: { color: n[3], width: 1.3 }, rectRadius: 0.06 });
    slide.addText(n[0], { x, y: 2.05, w: 2.0, h: 0.26, fontSize: 14, bold: true, color: C.text, align: 'center', margin: 0 });
    slide.addText(n[1], { x, y: 2.48, w: 2.0, h: 0.22, fontFace: 'Consolas', fontSize: 10.5, color: C.muted, align: 'center', margin: 0 });
    if (i < nodes.length - 1) addArrow(slide, x + 2.04, 2.43, x + 2.46, 2.43, C.muted);
  });
  addCard(slide, 0.8, 4.05, 3.6, 1.5, '1. Predict', 'Calculate expected bins, frames,\nshapes, and dtypes first.', C.paleBlue, C.blue);
  addCard(slide, 4.87, 4.05, 3.6, 1.5, '2. Implement', 'Build one transformation at a\ntime in the notebook.', C.paleGreen, C.green);
  addCard(slide, 8.93, 4.05, 3.6, 1.5, '3. Prove', 'Assert structural and semantic\ninvariants before moving on.', C.paleCoral, C.coral);
  slide.addText('Reflection questions', { x: 0.85, y: 5.95, w: 2.2, h: 0.32, fontSize: 15, bold: true, color: C.navy, margin: 0 });
  slide.addText('Why 513 bins?   Why 87 frames?   Why is 440 Hz not exactly one bin?   Why does mel preserve time frames?', {
    x: 2.7, y: 5.94, w: 9.7, h: 0.42, fontSize: 13, color: C.text, margin: 0, fit: 'shrink',
  });
}

const out = process.argv[2] || 'VITS2-Learning-Progress-Chapters-01-03.pptx';
pptx.writeFile({ fileName: out });

