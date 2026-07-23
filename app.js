import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const MODEL_URL = './model/curlew.glb';
const ILLUSTRATION_URL = './images/curlew-illustration.png';

const canvas = document.getElementById('scene');
const loaderEl = document.getElementById('loader');
const barFill = document.getElementById('bar-fill');
const statusEl = document.getElementById('loader-status');
const splash = document.getElementById('splash');
const splashPrompt = document.getElementById('splash-prompt');

/* ---------- renderer, scene, camera ---------- */

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, -2); // replaced by frameModel() once the model loads

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.rotateSpeed = 0.6;
controls.zoomSpeed = 0.8;
// The turntable holds off until the model has faded in, so the opening pose is
// the reference orientation rather than whatever the spin has drifted to.
controls.autoRotate = false;
controls.autoRotateSpeed = 0.7;

/* ---------- lighting ---------- */

const pmrem = new THREE.PMREMGenerator(renderer);
const envMap = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environment = envMap;
scene.environmentIntensity = 0.55;

const key = new THREE.DirectionalLight(0xfff4e2, 1.6);
key.position.set(2.5, 3, 2);
const fill = new THREE.DirectionalLight(0xbcd0ff, 0.5);
fill.position.set(-2.5, 0.5, -1.5);
const ambient = new THREE.AmbientLight(0xffffff, 0.3);
scene.add(key, fill, ambient);

/* ---------- model ---------- */

const state = {
  lit: true,
  wireframe: false,
  materials: { lit: [], unlit: [] },
  meshes: [],
  extent: new THREE.Vector3(1, 1, 1),
  revealed: false,
  fade: null,
  turntablePending: true,
  wrap: null,
  wrapStartZ: -0.35,
  wrapStartScale: 0.4,
  wrapPending: true, // the opening plays the plate on, rather than starting with it
  plate: null,
  specimen: null,
  projected: true, // the illustration is what the page is for; scan is the alternative
  artTween: null,
};

/* ---------- illustration projection ----------------------------------------

The plate is applied as a flat two-sided card along the view axis, so from the
front it reads as the drawing rather than as a distortion of it. An earlier
version unwrapped it cylindrically, which does close right around the specimen —
but a side-view drawing holds no detail at the back and belly, so the outline
smeared into swirls across both poles, and the default view looks straight at
one of them.

What a single flat card cannot do is place the head: it is turned and set at its
own angle to the body. So body and head are fitted separately and blended across
the neck — the body by centroid and principal axis, the head by its centre and
bill tip. The neck plane's normal runs body-to-head rather than straight up, so
the bill, which hangs below the neck in height, still counts as head.
--------------------------------------------------------------------------- */

const TAIL_TRIM = 0.08; // alpha mass trimmed off the plate's wispy tail tip
const BILL_TAU = 0.12; // column mass below this * peak is bill, not head
const NECK_TAU = 0.45; // ...and below this, scanning forward, is the neck
const INK_ALPHA = 8; // alpha above which a plate pixel counts as drawn
const BODY_TRIM = 3.0; // outlier cut in median radii, to drop the label tag
const NECK_SLICES = 48; // height slices used to find the neck
const NECK_SETBACK = 0.05; // height band below the neck used to place it
const NECK_BAND = 0.03; // half-width of the body-to-head blend
const WARP_SLICES = 64; // stations along each part where the warp is measured


const blank = new THREE.DataTexture(new Uint8Array([255, 255, 255, 0]), 1, 1);
blank.needsUpdate = true;

const projection = {
  texture: { value: blank },
  // per part: specimen origin + along axis, the along ranges, and the drawing's
  // origin + axes already divided by the plate size
  bodyAxis: { value: new THREE.Vector4(0, 0, 1, 0) },
  bodyRange: { value: new THREE.Vector4(0, 1, 0, 1) },
  bodyPlate: { value: new THREE.Vector4(0, 0, 1, 0) },
  bodyPlateN: { value: new THREE.Vector2(0, 1) },
  bodyLut: { value: blank },
  headAxis: { value: new THREE.Vector4(0, 0, 1, 0) },
  headRange: { value: new THREE.Vector4(0, 1, 0, 1) },
  headPlate: { value: new THREE.Vector4(0, 0, 1, 0) },
  headPlateN: { value: new THREE.Vector2(0, 1) },
  headLut: { value: blank },
  boneBody: { value: new THREE.Vector4(0, -1, 0, 0) }, // tail xy -> neck zw
  boneHead: { value: new THREE.Vector4(0, 0, 1, 0) }, // neck xy -> bill zw
  mix: { value: 0 },
};

function patchMaterial(material) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uArt = projection.texture;
    shader.uniforms.uBodyAxis = projection.bodyAxis;
    shader.uniforms.uBodyRange = projection.bodyRange;
    shader.uniforms.uBodyPlate = projection.bodyPlate;
    shader.uniforms.uBodyPlateN = projection.bodyPlateN;
    shader.uniforms.uBodyLut = projection.bodyLut;
    shader.uniforms.uHeadAxis = projection.headAxis;
    shader.uniforms.uHeadRange = projection.headRange;
    shader.uniforms.uHeadPlate = projection.headPlate;
    shader.uniforms.uHeadPlateN = projection.headPlateN;
    shader.uniforms.uHeadLut = projection.headLut;
    shader.uniforms.uBoneBody = projection.boneBody;
    shader.uniforms.uBoneHead = projection.boneHead;
    shader.uniforms.uArtMix = projection.mix;

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vProjected;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\n\tvProjected = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;',
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying vec3 vProjected;
        uniform sampler2D uArt;
        uniform vec4 uBodyAxis;
        uniform vec4 uBodyRange;
        uniform vec4 uBodyPlate;
        uniform vec2 uBodyPlateN;
        uniform sampler2D uBodyLut;
        uniform vec4 uHeadAxis;
        uniform vec4 uHeadRange;
        uniform vec4 uHeadPlate;
        uniform vec2 uHeadPlateN;
        uniform sampler2D uHeadLut;
        uniform vec4 uBoneBody;
        uniform vec4 uBoneHead;
        uniform float uArtMix;`,
      )
      .replace(
        '#include <common>',
        `#define NECK_BAND ${NECK_BAND.toFixed(4)}
        #define WARP_SLICES ${WARP_SLICES.toFixed(1)}
        vec2 warpPart( vec2 q, vec4 axis, vec4 rng, vec4 plate, vec2 plateN,
                       sampler2D lut ) {
          vec2 u = axis.zw;
          vec2 nv = vec2( -u.y, u.x );
          vec2 d = q - axis.xy;
          float a = dot( d, u );
          float c = dot( d, nv );

          float t = clamp( ( a - rng.x ) / max( rng.y - rng.x, 1e-6 ), 0.0, 1.0 );
          vec4 L = texture2D( lut,
            vec2( ( t * ( WARP_SLICES - 1.0 ) + 0.5 ) / WARP_SLICES, 0.5 ) );

          // where this point sits across the specimen, carried to the same
          // fraction across the drawing at the matching station
          float f = ( c - L.x ) / max( L.y - L.x, 1e-5 );
          float aP = rng.z + t * ( rng.w - rng.z );
          float cP = L.z + f * ( L.w - L.z );
          return plate.xy + aP * plate.zw + cP * plateN;
        }

        float segDist( vec2 p, vec2 a, vec2 b ) {
          vec2 v = b - a;
          vec2 w = p - a;
          float t = clamp( dot( w, v ) / max( dot( v, v ), 1e-8 ), 0.0, 1.0 );
          return length( w - t * v );
        }
        #include <common>`,
      )
      .replace(
        '#include <map_fragment>',
        `#include <map_fragment>
        {
          // the specimen faces screen-right, which is -x in world space
          vec2 q = vec2( -vProjected.x, vProjected.y );

          // whichever bone the point lies nearer decides which fit applies
          float dBody = segDist( q, uBoneBody.xy, uBoneBody.zw );
          float dHead = segDist( q, uBoneHead.xy, uBoneHead.zw );
          float w = smoothstep( -NECK_BAND, NECK_BAND, dBody - dHead );

          vec2 uvBody = warpPart( q, uBodyAxis, uBodyRange, uBodyPlate, uBodyPlateN, uBodyLut );
          vec2 uvHead = warpPart( q, uHeadAxis, uHeadRange, uHeadPlate, uHeadPlateN, uHeadLut );

          // Each warp only ever reads its own part of the drawing, so the drawn
          // head can no longer appear a second time on the shoulder.
          vec4 cBody = texture2D( uArt, uvBody );
          vec4 cHead = texture2D( uArt, uvHead );
          float inBody = step( 0.0, uvBody.x ) * step( uvBody.x, 1.0 )
                       * step( 0.0, uvBody.y ) * step( uvBody.y, 1.0 );
          float inHead = step( 0.0, uvHead.x ) * step( uvHead.x, 1.0 )
                       * step( 0.0, uvHead.y ) * step( uvHead.y, 1.0 );

          // Each fit keeps a floor of influence outside its own bone, so one can
          // fill where the other's drawing runs out instead of leaving bare skin.
          float floorW = 0.3;
          float wBody = cBody.a * inBody * ( 1.0 - w * ( 1.0 - floorW ) );
          float wHead = cHead.a * inHead * ( floorW + w * ( 1.0 - floorW ) );
          float total = min( wBody + wHead, 1.0 );
          vec3 art = ( cBody.rgb * wBody + cHead.rgb * wHead ) / max( total, 1e-4 );
          diffuseColor.rgb = mix( diffuseColor.rgb, art, total * uArtMix );
        }`,
      );
  };

  material.customProgramCacheKey = () => 'curlew-projection';
}

/** Distance from a point to the segment A-B. */
function distToSegment(x, y, A, B) {
  const vx = B[0] - A[0];
  const vy = B[1] - A[1];
  const wx = x - A[0];
  const wy = y - A[1];
  const len2 = vx * vx + vy * vy;
  const t = len2 > 0 ? Math.min(1, Math.max(0, (wx * vx + wy * vy) / len2)) : 0;
  return Math.hypot(wx - t * vx, wy - t * vy);
}

/**
 * Per-station silhouette profile of a point set in the A->B frame: how far the
 * shape reaches either side of its axis at each station along it. Warping one
 * profile onto another is what fills the specimen — a rigid fit can match the
 * drawing's overall proportions or its outline, but not both, and the specimen
 * is a very different shape from a standing bird.
 */
function profile(xs, ys, A, B) {
  let ux = B[0] - A[0];
  let uy = B[1] - A[1];
  const ulen = Math.hypot(ux, uy);
  ux /= ulen;
  uy /= ulen;

  const along = [];
  const across = [];
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - A[0];
    const dy = ys[i] - A[1];
    along.push(dx * ux + dy * uy);
    across.push(dx * -uy + dy * ux);
  }
  const sortedAlong = along.slice().sort((a, b) => a - b);
  const a0 = sortedAlong[Math.floor(sortedAlong.length * 0.005)];
  const a1 = sortedAlong[Math.floor(sortedAlong.length * 0.995)];

  const buckets = Array.from({ length: WARP_SLICES }, () => []);
  for (let i = 0; i < along.length; i++) {
    const t = (along[i] - a0) / (a1 - a0);
    if (t < 0 || t >= 1) continue;
    buckets[Math.min(WARP_SLICES - 1, Math.floor(t * WARP_SLICES))].push(across[i]);
  }

  const lo = new Float64Array(WARP_SLICES);
  const hi = new Float64Array(WARP_SLICES);
  const seen = new Uint8Array(WARP_SLICES);
  for (let i = 0; i < WARP_SLICES; i++) {
    const b = buckets[i];
    if (b.length < 12) continue;
    b.sort((p, q) => p - q);
    lo[i] = b[Math.floor(b.length * 0.015)];
    hi[i] = b[Math.floor(b.length * 0.985)];
    seen[i] = 1;
  }
  for (let i = 0; i < WARP_SLICES; i++) {
    if (seen[i]) continue;
    let a = i;
    let c = i;
    while (a >= 0 && !seen[a]) a--;
    while (c < WARP_SLICES && !seen[c]) c++;
    if (a < 0 && c >= WARP_SLICES) break;
    if (a < 0) {
      lo[i] = lo[c];
      hi[i] = hi[c];
    } else if (c >= WARP_SLICES) {
      lo[i] = lo[a];
      hi[i] = hi[a];
    } else {
      const f = (i - a) / (c - a);
      lo[i] = lo[a] + (lo[c] - lo[a]) * f;
      hi[i] = hi[a] + (hi[c] - hi[a]) * f;
    }
  }
  for (let pass = 0; pass < 2; pass++) {
    const l2 = lo.slice();
    const h2 = hi.slice();
    for (let i = 1; i < WARP_SLICES - 1; i++) {
      lo[i] = (l2[i - 1] + l2[i] + l2[i + 1]) / 3;
      hi[i] = (h2[i - 1] + h2[i] + h2[i + 1]) / 3;
    }
  }
  return { a0, a1, lo, hi, u: [ux, uy], n: [-uy, ux] };
}

/**
 * Column profile and landmarks of the plate, in pixels with y measured upwards.
 * The bird runs tail-left to bill-right; the bill is thin and the head thinner
 * than the body, so walking the column-mass profile finds tail, neck and bill.
 */
function plateProfile(image) {
  const canvas = document.createElement('canvas');
  const w = (canvas.width = image.naturalWidth || image.width);
  const h = (canvas.height = image.naturalHeight || image.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  const { data } = ctx.getImageData(0, 0, w, h);

  const alphaAt = (x, y) => data[(y * w + x) * 4 + 3];
  const mass = new Float64Array(w);
  let peak = 0;
  let peakAt = 0;
  let total = 0;
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const a = alphaAt(x, y);
      if (a > INK_ALPHA) mass[x] += a;
    }
    total += mass[x];
    if (mass[x] > peak) {
      peak = mass[x];
      peakAt = x;
    }
  }

  let acc = 0;
  let tailX = 0;
  for (let x = 0; x < w; x++) {
    acc += mass[x];
    if (acc >= total * TAIL_TRIM) {
      tailX = x;
      break;
    }
  }
  let billBase = w - 1;
  for (let x = w - 1; x >= 0; x--) {
    if (mass[x] > BILL_TAU * peak) {
      billBase = x;
      break;
    }
  }
  let neckX = peakAt;
  for (let x = peakAt; x <= billBase; x++) {
    if (mass[x] < NECK_TAU * peak) {
      neckX = x;
      break;
    }
  }

  // Alpha-weighted centroid and covariance of a band of columns.
  const moments = (x0, x1) => {
    let m = 0;
    let sx = 0;
    let sy = 0;
    for (let x = x0; x < x1; x++) {
      for (let y = 0; y < h; y++) {
        const a = alphaAt(x, y);
        if (a <= INK_ALPHA) continue;
        m += a;
        sx += a * x;
        sy += a * (h - 1 - y);
      }
    }
    const cx = sx / m;
    const cy = sy / m;
    let xx = 0;
    let xy = 0;
    let yy = 0;
    for (let x = x0; x < x1; x++) {
      for (let y = 0; y < h; y++) {
        const a = alphaAt(x, y);
        if (a <= INK_ALPHA) continue;
        const dx = x - cx;
        const dy = h - 1 - y - cy;
        xx += a * dx * dx;
        xy += a * dx * dy;
        yy += a * dy * dy;
      }
    }
    return { c: [cx, cy], cov: [xx / m, xy / m, yy / m] };
  };

  // Body outline: the longest unbroken ink run per column. The legs are drawn
  // detached below the belly and would otherwise inflate the measured width.
  const outX = [];
  const outY = [];
  const colCentre = new Float64Array(w).fill(-1);
  for (let x = 0; x < w; x++) {
    let best = -1;
    let bt = 0;
    let bb = -1;
    let start = -1;
    for (let y = 0; y <= h; y++) {
      const on = y < h && alphaAt(x, y) > INK_ALPHA;
      if (on) {
        if (start < 0) start = y;
      } else if (start >= 0) {
        if (y - 1 - start > best) {
          best = y - 1 - start;
          bt = start;
          bb = y - 1;
        }
        start = -1;
      }
    }
    if (bb < bt) continue;
    colCentre[x] = h - 1 - (bt + bb) / 2;
    for (let y = bt; y <= bb; y++) {
      outX.push(x);
      outY.push(h - 1 - y);
    }
  }

  let firstX = 0;
  let lastX = w - 1;
  for (let x = 0; x < w; x++) if (colCentre[x] >= 0) { firstX = x; break; }
  for (let x = w - 1; x >= 0; x--) if (colCentre[x] >= 0) { lastX = x; break; }

  const tail = [firstX, colCentre[firstX]];
  const neck = [neckX, colCentre[neckX] >= 0 ? colCentre[neckX] : h / 2];
  const bill = [lastX, colCentre[lastX]];

  const part = (from, to) => {
    const xs = [];
    const ys = [];
    for (let i = 0; i < outX.length; i++) {
      if (outX[i] >= from && outX[i] < to) {
        xs.push(outX[i]);
        ys.push(outY[i]);
      }
    }
    return { xs, ys };
  };
  const body = part(firstX, neckX);
  const head = part(neckX, w);

  return { width: w, height: h, body, head, tail, neck, bill, split: neckX / w };
}

const median = (arr) => {
  arr.sort((a, b) => a - b);
  return arr.length ? arr[arr.length >> 1] : 0;
};

/**
 * Landmarks of the specimen, in screen-space 2D: the plate is applied as a flat
 * two-sided card along the view axis, and the specimen faces screen-right, which
 * is -x in world space.
 */
function specimenProfile(meshes) {
  const px = [];
  const py = [];
  const v = new THREE.Vector3();
  for (const mesh of meshes) {
    const pos = mesh.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
      px.push(-v.x);
      py.push(v.y);
    }
  }
  const n = px.length;

  let y0 = Infinity;
  let y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    if (py[i] < y0) y0 = py[i];
    if (py[i] > y1) y1 = py[i];
  }

  // Width profile by height, to find the neck: the narrowest slice above the
  // widest. Only lightly smoothed — heavy smoothing swallows the head, which is
  // bent almost horizontal and spans little height.
  const width = new Float64Array(NECK_SLICES);
  const seen = new Uint8Array(NECK_SLICES);
  const bin = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    bin[i] = Math.min(
      NECK_SLICES - 1,
      Math.max(0, Math.floor(((py[i] - y0) / (y1 - y0)) * NECK_SLICES)),
    );
  }
  const byBin = Array.from({ length: NECK_SLICES }, () => []);
  for (let i = 0; i < n; i++) byBin[bin[i]].push(i);

  for (let s = 0; s < NECK_SLICES; s++) {
    const xs = byBin[s].map((i) => px[i]);
    const ys = byBin[s].map((i) => py[i]);
    if (xs.length < 30) continue;
    const mx = median(xs.slice());
    const my = median(ys.slice());
    const radii = xs.map((x, i) => Math.hypot(x - mx, ys[i] - my));
    const cut = 2 * median(radii.slice());
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < xs.length; i++) {
      if (Math.hypot(xs[i] - mx, ys[i] - my) < cut) {
        lo = Math.min(lo, xs[i]);
        hi = Math.max(hi, xs[i]);
      }
    }
    if (hi > lo) {
      width[s] = hi - lo;
      seen[s] = 1;
    }
  }
  for (let s = 0; s < NECK_SLICES; s++) {
    if (seen[s]) continue;
    let lo = s;
    let hi = s;
    while (lo >= 0 && !seen[lo]) lo--;
    while (hi < NECK_SLICES && !seen[hi]) hi++;
    width[s] = lo < 0 ? width[hi] : hi >= NECK_SLICES ? width[lo] : (width[lo] + width[hi]) / 2;
  }
  const smoothed = width.slice();
  for (let s = 1; s < NECK_SLICES - 1; s++) {
    smoothed[s] = (width[s - 1] + width[s] + width[s + 1]) / 3;
  }
  let widest = 0;
  for (let s = 0; s < NECK_SLICES; s++) if (smoothed[s] > smoothed[widest]) widest = s;
  let neckSlice = widest;
  const limit = Math.floor(NECK_SLICES * 0.92);
  for (let s = widest; s < limit; s++) if (smoothed[s] < smoothed[neckSlice]) neckSlice = s;
  const yNeck = y0 + ((neckSlice + 0.5) / NECK_SLICES) * (y1 - y0);

  // The bill tip is the specimen's extreme in the facing direction, taken over
  // the whole skin. It must NOT be looked for above the neck's height: the bill
  // droops, and half of it lies below that line — restricting the search finds a
  // point partway along the bill and squeezes the whole head fit into it.
  const allMx = median(px.slice());
  const allMy = median(py.slice());
  const cut = BODY_TRIM * median(px.map((x, i) => Math.hypot(x - allMx, py[i] - allMy)));
  let billX = -Infinity;
  let billY = 0;
  for (let i = 0; i < n; i++) {
    if (Math.hypot(px[i] - allMx, py[i] - allMy) >= cut) continue; // label tag
    if (px[i] > billX) {
      billX = px[i];
      billY = py[i];
    }
  }
  const bill = [billX, billY];

  // Neck: the skin's own centre in a thin band just below the head's height.
  const nx0 = [];
  const ny0 = [];
  for (let i = 0; i < n; i++) {
    if (py[i] > yNeck || py[i] <= yNeck - NECK_SETBACK) continue;
    if (Math.hypot(px[i] - allMx, py[i] - allMy) >= cut) continue;
    nx0.push(px[i]);
    ny0.push(py[i]);
  }
  const neck = [median(nx0.slice()), median(ny0.slice())];

  // A provisional tail, needed before the bones exist: the farthest point from
  // the neck that is not the bill.
  let seedTail = [neck[0], neck[1] - 0.5];
  let seedFar = 0;
  for (let i = 0; i < n; i++) {
    if (Math.hypot(px[i] - allMx, py[i] - allMy) >= cut) continue;
    if (py[i] > yNeck) continue;
    const dd = Math.hypot(px[i] - neck[0], py[i] - neck[1]);
    if (dd > seedFar) {
      seedFar = dd;
      seedTail = [px[i], py[i]];
    }
  }

  // Head and body are split by which of the two bones a point lies nearer —
  // tail-to-neck, or neck-to-bill. A dividing plane cannot do it: the bill
  // droops down alongside the body, so any plane that keeps the bill with the
  // head also takes a slice of the flank with it.
  const hx = [];
  const hy = [];
  const bx = [];
  const by = [];
  for (let i = 0; i < n; i++) {
    if (Math.hypot(px[i] - allMx, py[i] - allMy) >= cut) continue;
    if (distToSegment(px[i], py[i], neck, bill) < distToSegment(px[i], py[i], seedTail, neck)) {
      hx.push(px[i]);
      hy.push(py[i]);
    } else {
      bx.push(px[i]);
      by.push(py[i]);
    }
  }

  // Tail: the lowest part of the skin. Taking the body's farthest point from
  // the neck instead picks up the label tag, which hangs further out sideways
  // than the tail does downwards and survives the outlier cut.
  const lowest = by.slice().sort((a, b) => a - b)[Math.floor(by.length * 0.01)];
  const tx = [];
  const ty = [];
  for (let i = 0; i < bx.length; i++) {
    if (by[i] <= lowest) {
      tx.push(bx[i]);
      ty.push(by[i]);
    }
  }
  const tail = [median(tx.slice()), median(ty.slice())];

  return { tail, neck, bill, body: { xs: bx, ys: by }, head: { xs: hx, ys: hy } };
}

/**
 * Body and head are warped separately and blended across the neck.
 *
 * Each part takes its axis from anatomical landmarks that the drawing and the
 * specimen each state unambiguously — tail-to-neck, then neck-to-bill — and both
 * are then measured station by station along that axis. A point's position
 * across the specimen is carried to the same fraction across the drawing at the
 * matching station, so the drawing stretches to fill the specimen's outline.
 *
 * A rigid fit cannot do this. It can match the drawing's overall proportions or
 * its outline, but a standing bird and a flat-packed study skin are different
 * shapes, so whatever it matches, large areas fall outside the drawing and
 * revert to bare specimen. Warping takes coverage from ~64% to ~95%.
 */
function updateProjection() {
  if (!state.plate || !state.specimen) return;
  const plate = state.plate;
  const spec = state.specimen;
  const half = THREE.DataUtils.toHalfFloat;

  const makeLut = (data) => {
    const t = new THREE.DataTexture(data, WARP_SLICES, 1, THREE.RGBAFormat, THREE.HalfFloatType);
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.wrapS = THREE.ClampToEdgeWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  };

  const buildPart = (mA, mB, mPts, pA, pB, pPts, uniforms) => {
    const m = profile(mPts.xs, mPts.ys, mA, mB);
    const p = profile(pPts.xs, pPts.ys, pA, pB);

    const data = new Uint16Array(WARP_SLICES * 4);
    for (let i = 0; i < WARP_SLICES; i++) {
      data[i * 4] = half(m.lo[i]);
      data[i * 4 + 1] = half(m.hi[i]);
      data[i * 4 + 2] = half(p.lo[i]);
      data[i * 4 + 3] = half(p.hi[i]);
    }

    uniforms.axis.value.set(mA[0], mA[1], m.u[0], m.u[1]);
    uniforms.range.value.set(m.a0, m.a1, p.a0, p.a1);
    // fold the pixels-to-uv divide into the drawing's origin and axes
    uniforms.plate.value.set(
      pA[0] / plate.width,
      pA[1] / plate.height,
      p.u[0] / plate.width,
      p.u[1] / plate.height,
    );
    uniforms.plateN.value.set(p.n[0] / plate.width, p.n[1] / plate.height);
    uniforms.lut.value = makeLut(data);
  };

  buildPart(spec.tail, spec.neck, spec.body, plate.tail, plate.neck, plate.body, {
    axis: projection.bodyAxis,
    range: projection.bodyRange,
    plate: projection.bodyPlate,
    plateN: projection.bodyPlateN,
    lut: projection.bodyLut,
  });
  buildPart(spec.neck, spec.bill, spec.head, plate.neck, plate.bill, plate.head, {
    axis: projection.headAxis,
    range: projection.headRange,
    plate: projection.headPlate,
    plateN: projection.headPlateN,
    lut: projection.headLut,
  });

  projection.boneBody.value.set(spec.tail[0], spec.tail[1], spec.neck[0], spec.neck[1]);
  projection.boneHead.value.set(spec.neck[0], spec.neck[1], spec.bill[0], spec.bill[1]);

  // If the opening animation has already been and gone by the time the mapping
  // is ready, just show it.
  if (state.projected && !state.wrapPending && !state.wrap && !state.artTween) {
    projection.mix.value = 1;
  }
}

new THREE.TextureLoader().load(ILLUSTRATION_URL, (texture) => {
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  projection.texture.value = texture;
  state.plate = plateProfile(texture.image);
  updateProjection();
});

/* ---------- hint ---------- */

const hintEl = document.getElementById('hint');
let hintTimer = null;

/* ---------- model ---------- */

const gltfLoader = new GLTFLoader();

gltfLoader.load(MODEL_URL, onLoad, onProgress, onError);

function onProgress(event) {
  if (!event.lengthComputable || !event.total) return;
  const pct = Math.round((event.loaded / event.total) * 100);
  barFill.style.width = `${pct}%`;
  statusEl.textContent = `Loading model… ${pct}%`;
}

function onError(err) {
  console.error(err);
  statusEl.innerHTML =
    'Could not load the model.<br>Serve this folder over HTTP — e.g. <code>python3 -m http.server</code> — rather than opening the file directly.';
  statusEl.style.color = '#e8a0a0';
  splashPrompt.textContent = 'The 3D model could not be loaded';
}

function onLoad(gltf) {
  const model = gltf.scene;

  model.traverse((child) => {
    if (!child.isMesh) return;

    // Normals were dropped during decimation so the mesh could be welded;
    // recompute them smoothly here.
    child.geometry.computeVertexNormals();

    const map = child.material.map;
    if (map) map.anisotropy = renderer.capabilities.getMaxAnisotropy();

    const lit = new THREE.MeshStandardMaterial({ map, roughness: 0.92, metalness: 0.0 });
    const unlit = new THREE.MeshBasicMaterial({ map });
    patchMaterial(lit);
    patchMaterial(unlit);

    state.materials.lit.push(lit);
    state.materials.unlit.push(unlit);
    state.meshes.push(child);

    child.material = state.lit ? lit : unlit;
  });

  // The skin is scanned lying flat, head towards +Z. Stand it upright so the
  // turntable spins the specimen about its own long axis, then turn it to face
  // the camera: underside towards the viewer, bill to the right.
  model.rotation.x = -Math.PI / 2;
  model.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), Math.PI);

  // Scale so the longest dimension is 1, then centre on the origin. Both are
  // measured after the rotation, so the box is the one actually rendered.
  model.updateMatrixWorld(true);
  const size = new THREE.Box3().setFromObject(model).getSize(new THREE.Vector3());
  model.scale.setScalar(1 / Math.max(size.x, size.y, size.z));

  model.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(model);
  model.position.copy(box.getCenter(new THREE.Vector3())).negate();
  state.extent = box.getSize(new THREE.Vector3());

  model.updateMatrixWorld(true);
  state.specimen = specimenProfile(state.meshes);
  updateProjection();

  scene.add(model);

  frameModel();
  finishLoading();
}

function frameModel() {
  const vFov = (camera.fov * Math.PI) / 180;
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);

  // Fit the specimen's projected box rather than its bounding sphere — a sphere
  // around a long, flat study skin leaves most of the frame empty. This view is
  // also the widest the model ever gets, so the turntable cannot overflow it.
  const { x: width, y: height, z: depth } = state.extent;
  const distance =
    Math.max(height / 2 / Math.tan(vFov / 2), width / 2 / Math.tan(hFov / 2)) * 1.12 +
    depth / 2;

  controls.target.set(0, 0, 0);
  camera.position.set(0, 0, -distance);
  controls.minDistance = distance * 0.2;
  controls.maxDistance = distance * 3;
  controls.update();
}

function finishLoading() {
  barFill.style.width = '100%';
  loaderEl.classList.add('done');
  setTimeout(() => loaderEl.remove(), 700);

  splash.classList.add('ready');
  splashPrompt.textContent = 'Click to view the specimen in 3D';

  // If the visitor already dismissed the plate, the model arrives late — fade
  // it up now rather than popping it in.
  if (state.revealed) startModelFade();
}

/* ---------- splash plate ---------- */

function startModelFade() {
  if (!state.meshes.length) return;
  setModelOpacity(0);
  state.fade = { start: performance.now(), duration: 1600 };
}

function setModelOpacity(value) {
  const solid = value >= 1;
  for (const pool of [state.materials.lit, state.materials.unlit]) {
    for (const material of pool) {
      material.transparent = !solid;
      material.opacity = value;
      material.depthWrite = true;
    }
  }
}

function reveal() {
  if (state.revealed) return;
  state.revealed = true;

  document.body.classList.add('revealed');
  splash.classList.add('fading');
  splash.addEventListener('transitionend', () => (splash.hidden = true), { once: true });

  startModelFade();
  hintTimer = setTimeout(() => hintEl.classList.add('gone'), 7000);
}

splash.addEventListener('click', reveal);

/* ---------- ui ---------- */

function toggleRotate() {
  state.turntablePending = false;
  controls.autoRotate = !controls.autoRotate;
}

function toggleLighting() {
  state.lit = !state.lit;
  const pool = state.lit ? state.materials.lit : state.materials.unlit;
  state.meshes.forEach((mesh, i) => {
    pool[i].wireframe = state.wireframe;
    mesh.material = pool[i];
  });
}

function toggleWireframe() {
  state.wireframe = !state.wireframe;
  state.meshes.forEach((mesh) => (mesh.material.wireframe = state.wireframe));
}


/* ---------- the plate flying on ------------------------------------------

Switching to the illustration plays it as an object rather than a cross-fade: a
flat card of the drawing swings in front of the specimen, then rolls around it
and is absorbed. The roll happens in a vertex shader — the card's grid is blended
between a flat rectangle and a cylinder about the specimen's standing axis, with
each column of the grid starting slightly later than the one inside it, so it
curls from the spine outwards rather than snapping.
------------------------------------------------------------------------- */

const WRAP_DURATION = 2000; // ms
const WRAP_ARC = Math.PI * 2; // a full turn, so the card closes right around
const WRAP_CLEARANCE = 1.02; // radius, relative to the specimen's own cross-section
const WRAP_HEIGHT = 0.95; // ...and height, kept just short of full so it stays in frame

const wrapUniforms = {
  uArt: projection.texture,
  uProgress: { value: 0 },
  uOpacity: { value: 0 },
  uRadius: { value: 0.24 },
  uHeight: { value: 1 },
  uFlat: { value: new THREE.Vector3(0, 0, -0.35) },
  uFlatSize: { value: new THREE.Vector2(0.72, 1) },
  uFlatScale: { value: 0.85 },
  uTilt: { value: 0 },
};

const wrapCard = new THREE.Mesh(
  new THREE.PlaneGeometry(1, 1, 120, 120),
  new THREE.ShaderMaterial({
    uniforms: wrapUniforms,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: `
      uniform float uProgress;
      uniform float uRadius;
      uniform float uHeight;
      uniform vec3 uFlat;
      uniform vec2 uFlatSize;
      uniform float uFlatScale;
      uniform float uTilt;
      varying vec2 vUv;

      void main() {
        vUv = uv;

        // The card lying flat, swung about its own vertical axis so it arrives
        // at an angle and squares up as it lands. Note "flat" and "half" are
        // reserved words in GLSL ES and cannot be used as names here.
        vec2 corner = ( uv - 0.5 ) * uFlatSize * uFlatScale;
        float ct = cos( uTilt );
        float st = sin( uTilt );
        vec3 flatPos = uFlat + vec3( corner.x * ct, corner.y, corner.x * st );

        float ang = ( uv.x - 0.5 ) * ${WRAP_ARC.toFixed(5)};
        vec3 rolled = vec3( sin( ang ) * uRadius,
                            ( uv.y - 0.5 ) * uHeight,
                            -cos( ang ) * uRadius );

        // columns further from the spine start later, so the card curls
        float t = clamp( uProgress * 1.7 - abs( uv.x - 0.5 ) * 0.7, 0.0, 1.0 );
        t = t * t * ( 3.0 - 2.0 * t );

        gl_Position = projectionMatrix * modelViewMatrix * vec4( mix( flatPos, rolled, t ), 1.0 );
      }
    `,
    fragmentShader: `
      uniform sampler2D uArt;
      uniform float uOpacity;
      varying vec2 vUv;

      void main() {
        // the drawing runs tail-to-bill along the card's height, since the
        // specimen stands where the drawn bird walks
        vec4 art = texture2D( uArt, vec2( vUv.y, vUv.x ) );
        if ( art.a < 0.02 ) discard;
        gl_FragColor = vec4( art.rgb, art.a * uOpacity );
      }
    `,
  }),
);
wrapCard.visible = false;
wrapCard.renderOrder = 2;
// the vertex shader moves vertices well outside the plane's own bounds, so the
// geometry's bounding sphere says nothing about where the card actually is
wrapCard.frustumCulled = false;
scene.add(wrapCard);

const WRAP_START_DIST = 0.38; // where the card starts, as a fraction of the camera's distance
const WRAP_START_FILL = 0.85; // how much of the frame height it fills there
const WRAP_START_TILT = 0.55; // radians it is swung round on arrival

function playWrap() {
  if (!state.plate) return false;

  // Enclose the specimen rather than passing through it: the radius has to reach
  // the corner of its cross-section, not merely its half-width. An enclosing
  // cylinder is also nearer the lens than the specimen's own front face, so it
  // is kept a little short of full height to stay inside the frame.
  wrapUniforms.uRadius.value =
    Math.hypot(state.extent.x, state.extent.z) * 0.5 * WRAP_CLEARANCE;
  wrapUniforms.uHeight.value = state.extent.y * WRAP_HEIGHT;
  wrapUniforms.uFlatSize.value.set(
    state.extent.y * (state.plate.height / state.plate.width),
    state.extent.y,
  );

  // Start the card near the viewer and let it recede onto the specimen, so it
  // reads as flying in. It has to be scaled to suit: the camera sits barely more
  // than one model-height away, so a card of the specimen's own size close to
  // the lens overruns the frame several times over. Size it instead to fill a
  // set share of the frame at wherever it starts, and grow it as it recedes.
  const halfFov = Math.tan((camera.fov * Math.PI) / 360);
  const camDist = camera.position.length();
  const startDist = Math.max(camDist * WRAP_START_DIST, camera.near * 4);

  state.wrapStartZ = startDist - camDist;
  state.wrapStartScale =
    (WRAP_START_FILL * 2 * halfFov * startDist) / Math.max(state.extent.y, 1e-4);

  state.wrap = { start: performance.now() };
  wrapCard.visible = true;
  return true;
}

function updateWrap(now) {
  if (!state.wrap) return;
  const p = Math.min(1, (now - state.wrap.start) / WRAP_DURATION);
  const ease = (a, b) => {
    const t = Math.min(1, Math.max(0, (p - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };

  const approach = ease(0.1, 0.85);
  wrapUniforms.uProgress.value = ease(0.18, 0.88);
  wrapUniforms.uOpacity.value = ease(0, 0.12) * (1 - ease(0.82, 1));

  // slide in from the side and forward, squaring up and growing as it lands
  const endZ = -wrapUniforms.uRadius.value;
  wrapUniforms.uFlat.value.set(
    -state.extent.x * 0.45 * (1 - approach),
    0,
    state.wrapStartZ + (endZ - state.wrapStartZ) * approach,
  );
  wrapUniforms.uFlatScale.value =
    state.wrapStartScale + (1 - state.wrapStartScale) * approach;
  wrapUniforms.uTilt.value = WRAP_START_TILT * (1 - approach);

  // the specimen takes the drawing on as the card is absorbed
  projection.mix.value = Math.max(projection.mix.value, ease(0.66, 1));

  if (p >= 1) {
    state.wrap = null;
    wrapCard.visible = false;
    projection.mix.value = 1;
  }
}

const textureButtons = [...document.querySelectorAll('#texture button')];

function syncTextureButtons() {
  for (const b of textureButtons) {
    const on = (b.dataset.mode === 'illustration') === state.projected;
    b.classList.toggle('on', on);
    b.setAttribute('aria-checked', String(on));
  }
}

function setTexture(showIllustration, animate = true) {
  state.projected = showIllustration;
  state.wrapPending = false;
  syncTextureButtons();
  if (!animate) {
    state.artTween = null;
    state.wrap = null;
    wrapCard.visible = false;
    projection.mix.value = showIllustration ? 1 : 0;
    return;
  }

  // The flourish belongs to the opening only. Once the visitor is driving the
  // toggle they want to compare the two surfaces, and a two-second animation
  // every time gets in the way of that.
  state.wrap = null;
  wrapCard.visible = false;
  state.artTween = {
    from: projection.mix.value,
    to: showIllustration ? 1 : 0,
    start: performance.now(),
    duration: 500,
  };
}

for (const b of textureButtons) {
  b.addEventListener('click', () => setTexture(b.dataset.mode === 'illustration'));
}

syncTextureButtons(); // markup and state cannot drift apart

function toggleProjection() {
  setTexture(!state.projected);
}


function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen();
}

const infoPanel = document.getElementById('info');
const infoToggle = document.getElementById('info-toggle');

infoToggle.addEventListener('click', () => {
  infoPanel.hidden = !infoPanel.hidden;
  infoToggle.setAttribute('aria-expanded', String(!infoPanel.hidden));
  infoToggle.classList.toggle('active', !infoPanel.hidden);
});

window.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;

  // While the plate is up, the only control is "dismiss it".
  if (!state.revealed) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      reveal();
    }
    return;
  }

  switch (event.key.toLowerCase()) {
    case ' ':
      event.preventDefault();
      toggleRotate();
      break;
    case 'l':
      toggleLighting();
      break;
    case 'w':
      toggleWireframe();
      break;
    case 'p':
      toggleProjection();
      break;
    case 'f':
      toggleFullscreen();
      break;
    case 'r':
      frameModel();
      break;
    case 'escape':
      if (!infoPanel.hidden) infoToggle.click();
      break;
  }
});

// Stop the turntable as soon as the visitor takes hold of the model.
controls.addEventListener('start', () => {
  state.turntablePending = false;
  controls.autoRotate = false;
});

/* ---------- loop ---------- */

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const smoothstep = (t) => t * t * (3 - 2 * t);

renderer.setAnimationLoop(() => {
  updateWrap(performance.now());

  if (state.artTween) {
    const t = (performance.now() - state.artTween.start) / state.artTween.duration;
    if (t >= 1) {
      projection.mix.value = state.artTween.to;
      state.artTween = null;
    } else {
      const { from, to } = state.artTween;
      projection.mix.value = from + (to - from) * smoothstep(t);
    }
  }

  if (state.fade) {
    const t = (performance.now() - state.fade.start) / state.fade.duration;
    if (t >= 1) {
      setModelOpacity(1);
      state.fade = null;
      if (state.turntablePending) {
        state.turntablePending = false;
        controls.autoRotate = true;
      }
      if (state.wrapPending) {
        state.wrapPending = false;
        // if the drawing is not ready yet, updateProjection will show it
        if (state.projected && playWrap()) projection.mix.value = 0;
      }
    } else {
      setModelOpacity(smoothstep(t));
    }
  }

  controls.update();
  renderer.render(scene, camera);
});
