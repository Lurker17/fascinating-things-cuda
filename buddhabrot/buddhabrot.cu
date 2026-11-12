// =====================================================================================
//  Divine Buddhabrot — a CUDA animation
// =====================================================================================
//
//  THE IDEA (in one breath)
//  ------------------------
//  The Mandelbrot iteration is   z -> z² + c,   starting from z = 0.
//  For most points c outside the Mandelbrot set, z eventually flies off to infinity.
//  The Buddhabrot does NOT colour c. Instead, for every escaping c, it draws the whole
//  path z₁, z₂, z₃ ... that z took on its way out, and counts how many paths cross
//  each pixel. The density of all those escape paths looks like a seated Buddha.
//
//  COLOUR ("moonlight", like the classic Buddhabrot on Wikipedia)
//  ------------------------------------------------------------------
//  We keep three counts per pixel, split by how long each path took to escape:
//      fast   escapers ( 10 –   60 steps)  -> a broad haze
//      medium escapers ( 60 –  600 steps)  -> the body
//      slow   escapers (600 – 6000 steps)  -> the fine filaments
//  They are mixed into one density, with the haze turned down so the hollows of the
//  head and belly stay dark: the contrast between dark hollows and bright rims is
//  what makes the figure look carved out of light.
//  The density is then coloured by raising it to a different power per channel:
//      red = d^1.6,  green = d^1.25,  blue = d^0.8
//  Blue rises first, so faint light is deep blue, brighter light icy blue, and only
//  the densest rims turn white. Empty space stays black.
//
//  LIGHT (simple tricks that make it feel "divine")
//  ------------------------------------------------
//    1. tight glow  - a slightly blurred copy added on top: bright edges shimmer
//    2. wide glow   - a heavily blurred copy: the soft aura around the figure
//    3. light rays  - every pixel looks back toward the heart of the figure and
//                     gathers the bright light it passes; bright parts streak outward
//                     like sunbeams behind a statue (a "radial blur")
//    4. behind, not over - the aura and rays sit BEHIND the figure: they fade out
//                     wherever the figure itself is bright, so the face stays crisp
//    5. nimbus      - a soft silver ring behind the head, like the halo in Buddhist
//                     iconography. Head paths sit near z = -1 with c = -1, so through
//                     the 4D camera the head is at  -cos θ - sin θ. It fades as we turn.
//    6. breathing   - the glow and rays swell and fade gently, like slow breath
//    7. auto-exposure - like a camera: dim scenes are brightened to a steady level
//    8. aureole     - 48 thin rays of light fanning out from behind the halo, long and
//                     short alternating, like the carved sunburst behind temple statues
//    9. third eye   - a small jewel of light on the brow (the urna), with a star glint
//   10. rising motes - tiny motes of light drifting upward around the figure,
//                     like sparks from incense
//  (5, 8, 9, 10 belong to the Buddha: they fade away as it turns into the Mandelbrot set)
//  Finally a film curve rolls the brightest light off gently into white.
//
//  THE UNIVERSE (background)
//  -------------------------
//  We fly slowly forward through space, toward the Buddha at the centre.
//    stars     - one random star per grid cell, with random brightness and colour.
//                The "infinite zoom": 4 copies of the starfield at doubling sizes.
//                Each copy grows, fading in while tiny and out when huge; after one
//                loop every copy has become exactly the next one. Endless flight,
//                seamless loop. Near (large) stars are brighter and some sparkle.
//    galaxies  - rare, faint, tilted ellipses riding along with the stars
//    nebulae   - "fractal noise" (smooth noise added at doubling detail levels gives
//                cloud shapes), coloured violet/teal, zooming the same way
//    milky way - a diagonal band where stars and clouds are denser
//
//  ANIMATION (one knob: an angle θ)
//  -------------------------------
//  Every point on a path really lives in 4D:  (Re z, Im z, Re c, Im c).
//  Normally we look at the (Re z, Im z) plane  -> the Buddha.
//  If we look at the (Re c, Im c) plane instead -> the ordinary Mandelbrot set
//  (every path point lands back on its own c, so we see where the c's are).
//  We turn the camera between the two:
//        screen_vertical   = Re z · cos θ + Re c · sin θ
//        screen_horizontal = Im z · cos θ + Im c · sin θ
//  The loop: θ flows smoothly 0° -> 90° -> 0° in one continuous wave (no pauses).
//
//  GPU STRATEGY (why it's fast)
//  ----------------------------
//  The camera moves, but the paths themselves never change. So:
//    once:        throw ~268M random c's, keep only those whose paths we'll draw
//                 (escape within a colour band), and SORT them by path length.
//    every frame: one thread per kept c replays its path and adds +1 to each pixel.
//  Sorting matters because GPU threads run in lockstep groups of 32: if they all
//  have similar path lengths, nobody waits for a slow neighbour.
//
//  BUILD / RUN
//  -----------
//    nvcc -O3 -arch=sm_80 -o buddhabrot buddhabrot.cu
//    ./buddhabrot W H frames [start end log2samples] | ffmpeg -f rawvideo -pix_fmt rgb24 \
//        -s WxH -r 30 -i - -c:v libx264 -crf 16 -pix_fmt yuv420p out.mp4
//    (or just ./render.sh)
// =====================================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---- Colour bands: [min, max] escape-step counts --------------------------------------
constexpr int NBANDS = 3;
__constant__ int BAND_MIN[NBANDS] = { 10,  60,  600};
__constant__ int BAND_MAX[NBANDS] = { 60, 600, 6000};
constexpr int MIN_STEPS = 10, MAX_ITER = 6000;

struct Camera {
    int   W, H;
    float cosT, sinT;   // the 4D viewing angle θ
    float center;       // value of the vertical axis at screen centre
    float zoom;         // pixels per unit
};

// Small integer hash -> pseudo-random bits (PCG). Same input => same output.
__device__ __forceinline__ uint32_t hash(uint32_t v) {
    uint32_t s = v * 747796405u + 2891336453u;
    uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// One step of  z -> z² + c.
// The _rn intrinsics forbid the compiler from fusing operations differently in the two
// places we iterate. Otherwise long, chaotic paths could replay differently than measured.
__device__ __forceinline__ void step(float& zr, float& zi, float cr, float ci) {
    float r = __fadd_rn(__fsub_rn(__fmul_rn(zr, zr), __fmul_rn(zi, zi)), cr);
    float i = __fadd_rn(__fmul_rn(2.f * zr, zi), ci);
    zr = r; zi = i;
}

// ---- ONCE: find the c's worth drawing ------------------------------------------------
__global__ void findPaths(uint64_t firstSample, float2* keptC, int* keptN, unsigned* keptCount) {
    uint64_t id = firstSample + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // random c in the box [-2.05, 0.55] x [0, 1.25] (top half; the bottom is its mirror)
    uint32_t h1 = hash((uint32_t)id ^ hash((uint32_t)(id >> 32) + 0x9e3779b9u));
    uint32_t h2 = hash(h1 ^ 0x85ebca6bu);
    float cr = -2.05f + 2.60f * (h1 * 2.3283064e-10f);
    float ci =          1.25f * (h2 * 2.3283064e-10f);

    // skip the two big blobs of the Mandelbrot set: they never escape
    float q = (cr - 0.25f) * (cr - 0.25f) + ci * ci;
    if (q * (q + (cr - 0.25f)) <= 0.25f * ci * ci) return;      // main cardioid
    if ((cr + 1.f) * (cr + 1.f) + ci * ci <= 0.0625f) return;   // period-2 disc

    // count steps until |z| > 2. Trapped points fall into a repeating cycle: we remember
    // z every 16, 32, 64 ... steps, and if z lands exactly on it again we stop early.
    float zr = 0.f, zi = 0.f, savedR = 0.f, savedI = 0.f;
    int n = 0, nextSave = 16;
    while (n < MAX_ITER && zr * zr + zi * zi <= 4.f) {
        step(zr, zi, cr, ci); ++n;
        if (zr == savedR && zi == savedI) return;                // cycle: never escapes
        if (n == nextSave) { savedR = zr; savedI = zi; nextSave *= 2; }
    }
    if (n < MIN_STEPS || n >= MAX_ITER) return;                  // too quick, or trapped

    unsigned slot = atomicAdd(keptCount, 1u);
    keptC[slot] = make_float2(cr, ci);
    keptN[slot] = n;
}

// ---- EVERY FRAME: replay each kept path through the 4D camera --------------------------
__global__ void drawPaths(Camera cam, const float2* keptC, const int* keptN, unsigned count,
                          uint32_t* hist) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float cr = keptC[idx].x, ci = keptC[idx].y;
    int n = keptN[idx];

    bool inBand[NBANDS];
    #pragma unroll
    for (int b = 0; b < NBANDS; ++b) inBand[b] = n >= BAND_MIN[b] && n <= BAND_MAX[b];

    const size_t plane = (size_t)cam.W * cam.H;
    const float crs = cr * cam.sinT, cis = ci * cam.sinT;
    float zr = 0.f, zi = 0.f;
    for (int k = 0; k < n; ++k) {
        step(zr, zi, cr, ci);
        float vert  = zr * cam.cosT + crs;      // the 4D -> 2D camera
        float horiz = zi * cam.cosT + cis;
        int py  = (int)floorf((vert - cam.center) * cam.zoom + 0.5f * cam.H);
        int px  = (int)floorf( horiz * cam.zoom + 0.5f * cam.W);
        int pxm = (int)floorf(-horiz * cam.zoom + 0.5f * cam.W);  // mirror image (path of conj(c))
        if (py < 0 || py >= cam.H) continue;
        #pragma unroll
        for (int b = 0; b < NBANDS; ++b) if (inBand[b]) {
            uint32_t* row = hist + b * plane + (size_t)py * cam.W;
            if (px  >= 0 && px  < cam.W) atomicAdd(&row[px],  1u);
            if (pxm >= 0 && pxm < cam.W) atomicAdd(&row[pxm], 1u);
        }
    }
}

// Counts -> light ("moonlight", like the classic Wikipedia Buddhabrot).
// 1. density: each band's count divided by its "full brightness" level, then mixed.
//    The fast-escaper band gets a lower weight: it is a broad haze that would fill the
//    dark hollows of the head and belly, and those hollows are what give the figure depth.
// 2. colour ramp: the same density raised to a different power per channel. Blue rises
//    first, green next, red last, so faint light is deep blue, brighter light icy blue,
//    and only the densest rims turn white. Empty space stays black.
// Also writes `warm` = density, which feeds the light rays.
__global__ void countsToLight(const uint32_t* hist, float3* light, float* warm, int N,
                              float3 invNorm, float3 weights, float exposure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = exposure * (weights.x * hist[i]         * invNorm.x +
                          weights.y * hist[N + i]     * invNorm.y +
                          weights.z * hist[2 * N + i] * invNorm.z);
    float u = fminf(v, 1.f);
    light[i] = make_float3(powf(u, 1.6f), powf(u, 1.25f), powf(fminf(1.1f * v, 1.f), 0.8f));
    warm[i] = v;
}

// Light rays: walk from this pixel toward the light centre, averaging the warm light we
// pass over (nearer samples count more). Bright structure behind us smears outward = beams.
// Only light within `sourceRadius` of the centre casts rays (around the head, for the Buddha).
__global__ void lightRays(const float* warm, float3* rays, int W, int H, float cx, float cy,
                          float sourceRadius, float threshold, float3 tint, uint32_t frame) {
    const int   STEPS = 64;
    const float REACH = 0.35f;   // look back 35% of the way to the centre
    const float DECAY = 0.975f;  // each step further back counts a little less
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float jitter = hash(y * W + x + frame * 7919u) * 2.3283064e-10f;   // hides stepping bands
    float dx = (cx - x) * REACH / STEPS, dy = (cy - y) * REACH / STEPS;
    float sum = 0.f, wsum = 0.f, w = 1.f;
    for (int k = 0; k < STEPS; ++k, w *= DECAY) {
        int sx = (int)(x + dx * (k + jitter)), sy = (int)(y + dy * (k + jitter));
        float d2 = ((sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)) / (sourceRadius * sourceRadius);
        sum  += w * fmaxf(warm[sy * W + sx] - threshold, 0.f) * expf(-d2);
        wsum += w;
    }
    float v = sum / wsum;
    rays[y * W + x] = make_float3(v * tint.x, v * tint.y, v * tint.z);
}

// Gaussian blur along one axis (run twice: horizontal, then vertical) -> the glows.
__global__ void blur1D(const float3* src, float3* dst, int W, int H, bool horizontal,
                       const float* weights, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float3 acc = make_float3(0.f, 0.f, 0.f);
    for (int k = -radius; k <= radius; ++k) {
        int sx = horizontal ? min(max(x + k, 0), W - 1) : x;
        int sy = horizontal ? y : min(max(y + k, 0), H - 1);
        float3 v = src[sy * W + sx];
        float w = weights[abs(k)];
        acc.x += w * v.x; acc.y += w * v.y; acc.z += w * v.z;
    }
    dst[y * W + x] = acc;
}

// ==== THE UNIVERSE ====================================================================
__device__ __forceinline__ float rand01(int a, int b, uint32_t seed) {
    return hash((uint32_t)a * 73856093u ^ (uint32_t)b * 19349663u ^ seed * 83492791u) * 2.3283064e-10f;
}

// Smooth random hills: random heights on a grid, blended smoothly in between.
__device__ float valueNoise(float x, float y, uint32_t seed) {
    float fx = floorf(x), fy = floorf(y);
    int xi = (int)fx, yi = (int)fy;
    float u = x - fx, v = y - fy;
    u = u * u * (3.f - 2.f * u); v = v * v * (3.f - 2.f * v);
    float a = rand01(xi, yi, seed),     b = rand01(xi + 1, yi, seed);
    float c = rand01(xi, yi + 1, seed), d = rand01(xi + 1, yi + 1, seed);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

// Fractal noise: the same noise at doubling detail and halving strength -> clouds.
__device__ float fbm(float x, float y, uint32_t seed) {
    float sum = 0.f, amp = 0.5f;
    for (int o = 0; o < 6; ++o) {
        sum += amp * valueNoise(x, y, seed + o);
        float nx = 1.6f * x + 1.2f * y, ny = -1.2f * x + 1.6f * y;   // rotate + double
        x = nx + 3.1f; y = ny - 1.7f; amp *= 0.5f;
    }
    return sum;
}

// One copy of the starfield (plus galaxies). q = position in grid cells; cellPx = cell size
// in pixels; near = 0 (far, tiny) .. 1 (close, large); band = milky-way strength here.
__device__ float3 starLayer(float qx, float qy, float cellPx, float near, float band, float t) {
    float3 col = make_float3(0.f, 0.f, 0.f);
    int cx = (int)floorf(qx), cy = (int)floorf(qy);
    for (int j = -1; j <= 1; ++j) for (int i = -1; i <= 1; ++i) {
        int gx = cx + i, gy = cy + j;
        if (rand01(gx, gy, 11) > 0.35f + 0.60f * band) continue;          // empty cell
        float sx = gx + rand01(gx, gy, 12), sy = gy + rand01(gx, gy, 13);  // star position
        float dx = (qx - sx) * cellPx, dy = (qy - sy) * cellPx;            // in pixels
        float r = rand01(gx, gy, 14);
        float bright = (0.07f + 2.0f * r * r * r * r * r * r * r * r) * (0.4f + 1.2f * near);
        float twinkle = 0.8f + 0.2f * sinf(6.2831853f * (3.f * t + rand01(gx, gy, 15)));
        float sigma = 0.55f + 0.9f * near;
        float glow = expf(-(dx * dx + dy * dy) / (2.f * sigma * sigma));
        if (bright > 1.4f)                                                 // sparkle cross (rare)
            glow += 0.2f * (expf(-fabsf(dx) / (6.f * sigma)) * expf(-dy * dy / (0.5f * sigma * sigma)) +
                             expf(-fabsf(dy) / (6.f * sigma)) * expf(-dx * dx / (0.5f * sigma * sigma)));
        float temp = rand01(gx, gy, 16);                                   // blue-white .. gold
        float3 tint = temp < 0.5f ? make_float3(0.75f + 0.5f * temp, 0.85f + 0.2f * temp, 1.f)
                                  : make_float3(1.f, 1.05f - 0.3f * temp, 1.25f - 0.75f * temp);
        float v = bright * twinkle * glow;
        col.x += v * tint.x; col.y += v * tint.y; col.z += v * tint.z;
    }
    // galaxies: a much coarser grid, rarely occupied
    const float G = 9.f;
    int gx = (int)floorf(qx / G), gy = (int)floorf(qy / G);
    if (rand01(gx, gy, 21) < 0.18f) {
        float px = (gx + 0.2f + 0.6f * rand01(gx, gy, 22)) * G, py = (gy + 0.2f + 0.6f * rand01(gx, gy, 23)) * G;
        float ang = 6.2831853f * rand01(gx, gy, 24), tilt = 0.25f + 0.5f * rand01(gx, gy, 25);
        float dx = (qx - px) * cellPx, dy = (qy - py) * cellPx;
        float a =  cosf(ang) * dx + sinf(ang) * dy, b = (-sinf(ang) * dx + cosf(ang) * dy) / tilt;
        float size = cellPx * (0.5f + 0.8f * rand01(gx, gy, 26));
        float d = sqrtf(a * a + b * b) / size;
        float v = (0.5f * expf(-d * d * 6.f) + 0.25f * expf(-d * 2.5f)) * 0.06f * (0.5f + near);
        bool warmG = rand01(gx, gy, 27) < 0.5f;
        col.x += v * (warmG ? 1.0f : 0.7f); col.y += v * (warmG ? 0.85f : 0.8f); col.z += v * (warmG ? 0.6f : 1.0f);
    }
    return col;
}

// Nebula clouds at position (x, y) in "sky units".
__device__ float3 nebula(float x, float y, float band) {
    float n1 = fbm(1.8f * x, 1.8f * y, 101);
    float n2 = fbm(2.5f * x + 7.f, 2.5f * y - 3.f, 202);
    float dust = fbm(3.5f * x - 2.f, 3.5f * y + 5.f, 303);
    float cloud = fmaxf(n1 - 0.38f, 0.f); cloud = cloud * cloud * 6.f;
    cloud *= (0.35f + 1.2f * band) * (1.f - 0.6f * fmaxf(dust - 0.45f, 0.f) * 4.f);   // dark dust lanes
    float3 violet = make_float3(0.32f, 0.12f, 0.55f), teal = make_float3(0.06f, 0.26f, 0.45f);
    float3 rose = make_float3(0.45f, 0.12f, 0.25f);
    float m = fminf(fmaxf(n2 * 1.6f - 0.3f, 0.f), 1.f);
    float3 c = make_float3(violet.x + (teal.x - violet.x) * m, violet.y + (teal.y - violet.y) * m,
                           violet.z + (teal.z - violet.z) * m);
    float r = fmaxf(n2 - 0.6f, 0.f) * 3.f;
    c.x += rose.x * r; c.y += rose.y * r; c.z += rose.z * r;
    return make_float3(c.x * cloud * 0.5f, c.y * cloud * 0.5f, c.z * cloud * 0.5f);
}

// The whole sky for one frame. t = loop time in [0,1). (cx, cy) = where we fly toward.
__global__ void universe(float3* sky, int W, int H, float t, float cx, float cy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float ux = (x - cx) / H, uy = (y - cy) / H;           // sky units, centred on the flight target

    // milky way: a tilted band through the centre
    const float BAND_ANGLE = -0.45f;
    float across = -sinf(BAND_ANGLE) * ux + cosf(BAND_ANGLE) * uy;
    float band = expf(-across * across / (0.22f * 0.22f));

    // deep-space base colour: dark navy, a little indigo light behind the figure
    float glow = expf(-2.5f * (ux * ux + uy * uy));
    float3 c = make_float3(0.003f + 0.008f * glow + 0.010f * band, 0.003f + 0.005f * glow + 0.008f * band,
                           0.010f + 0.018f * glow + 0.014f * band);

    // stars + galaxies: 4 copies at doubling sizes, each growing 2x per loop (infinite zoom)
    const int   LAYERS = 4;
    const float CELLS  = 55.f;                           // grid cells per screen height, smallest copy
    for (int k = 0; k < LAYERS; ++k) {
        float phase = (k + t) / LAYERS;                  // 0 = far & tiny -> 1 = near & huge
        float zoom  = exp2f(LAYERS * phase);             // 1 .. 16
        float fade  = sinf(3.14159265f * phase); fade *= fade;
        float spin  = 0.5f * phase;                      // a slow swirl as we fly
        float rx = cosf(spin) * ux - sinf(spin) * uy, ry = sinf(spin) * ux + cosf(spin) * uy;
        float cellPx = H * zoom / CELLS;
        float3 s = starLayer(rx * CELLS / zoom, ry * CELLS / zoom, cellPx, phase, band, t);
        c.x += fade * s.x; c.y += fade * s.y; c.z += fade * s.z;
    }

    // nebulae: 2 copies, same trick (their fades always add up to 1)
    for (int k = 0; k < 2; ++k) {
        float phase = (k + t) / 2.f;
        float zoom  = exp2f(2.f * phase);
        float fade  = sinf(3.14159265f * phase); fade *= fade;
        float3 n = nebula(ux / zoom + 3.7f * k, uy / zoom - 1.3f * k, band);
        c.x += fade * n.x; c.y += fade * n.y; c.z += fade * n.z;
    }
    sky[y * W + x] = c;
}

// Film-like curve: bright values roll off softly to white instead of clipping.
__device__ __forceinline__ float filmic(float x) {
    x = fmaxf(x, 0.f);
    return fminf((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 1.f);
}

struct Mix { float exposure, tight, wide, rays; };

struct Backdrop {
    float  occlusion;                   // how strongly the figure hides what's behind it
    float  nimbusX, nimbusY, nimbusR;   // halo ring centre and radius (pixels)
    float  nimbusWidth, nimbusStrength;
    float3 nimbusColor;
    float  sunburst;                    // strength of the aureole of thin rays
    float  jewel;                       // strength of the third-eye jewel
    float  jewelY;                      // its height (pixels)
    float  motes;                       // strength of the rising motes of light
    float  t;                           // loop time in [0, 1)
    float  skyLevel;                    // brightness of the universe
};

// ==== THE SACRED LIGHT (only while we see the Buddha) ===================================
__device__ __forceinline__ float smoothstepf(float e0, float e1, float x) {
    float u = fminf(fmaxf((x - e0) / (e1 - e0), 0.f), 1.f);
    return u * u * (3.f - 2.f * u);
}

// Aureole: thin rays of light fanning out from behind the head, long and short alternating,
// as in the carved sunbursts behind temple statues. Turns by exactly two rays per loop
// (so the loop is seamless); a slow wave of brightness travels around it.
__device__ float aureole(float x, float y, const Backdrop& bd, float H) {
    const int RAYS = 48;
    float dx = x - bd.nimbusX, dy = y - bd.nimbusY;
    float r = sqrtf(dx * dx + dy * dy);
    float a = atan2f(dy, dx);
    float u = a * (RAYS / 6.2831853f) + 2.f * bd.t;        // position in "ray units"
    float k = floorf(u + 0.5f), d = u - k;                 // nearest ray, offset from it
    bool  longRay = ((int)k & 1) == 0;
    float line = expf(-d * d / (2.f * 0.045f * 0.045f));    // a thin line
    float reach = (longRay ? 0.40f : 0.22f) * H;
    float start = smoothstepf(0.9f * bd.nimbusR, 1.15f * bd.nimbusR, r);  // begins at the halo
    float fall  = expf(-(r - bd.nimbusR) / reach);
    float wave  = 0.65f + 0.35f * cosf(3.f * a - 6.2831853f * bd.t);
    return line * start * fall * wave;
}

// Third-eye jewel: a point of light with a four-pointed glint, like a star.
__device__ float jewelGlow(float x, float y, float cx, float cy, float H) {
    float dx = (x - cx) / H * 1080.f, dy = (y - cy) / H * 1080.f;   // in 1080p pixels
    float core  = expf(-(dx * dx + dy * dy) / (2.f * 1.8f * 1.8f));
    float bloom = 0.25f * expf(-sqrtf(dx * dx + dy * dy) / 9.f);
    float glint = expf(-fabsf(dx) / 11.f) * expf(-dy * dy / 1.5f)
                + expf(-fabsf(dy) / 11.f) * expf(-dx * dx / 1.5f);
    return core + bloom + 0.6f * glint;
}

// Rising motes: tiny motes of light drifting upward around the figure, like incense
// sparks. One possible mote per grid cell; the grid rises a whole number of cells per
// loop, so the last frame flows into the first.
__device__ float risingMotes(float x, float y, float cx, float cy, float W, float H, float t) {
    const float CELLS = 36.f;                 // cells per screen height
    const int   RISE  = 6;                    // cells risen per loop
    float qx = x / H * CELLS, qy = y / H * CELLS + RISE * t;
    int gx = (int)floorf(qx), gy0 = (int)floorf(qy);
    float sum = 0.f;
    for (int j = -1; j <= 1; ++j) for (int i = -1; i <= 1; ++i) {
        int gx2 = gx + i, gy = gy0 + j;
        if (rand01(gx2, gy, 51) > 0.14f) continue;
        float sway = 0.25f * sinf(6.2831853f * (2.f * t + rand01(gx2, gy, 52)));
        float sx = gx2 + 0.5f + 0.4f * (rand01(gx2, gy, 53) - 0.5f) + sway;
        float sy = gy + rand01(gx2, gy, 54);
        float ddx = (qx - sx) * H / CELLS, ddy = (qy - sy) * H / CELLS;  // pixels
        float size = (0.8f + 1.4f * rand01(gx2, gy, 55)) * H / 1080.f;
        float flicker = 0.5f + 0.5f * sinf(6.2831853f * (4.f * t + rand01(gx2, gy, 56)));
        sum += (0.3f + 0.7f * flicker) * expf(-(ddx * ddx + ddy * ddy) / (2.f * size * size));
    }
    // only near the figure: a soft oval around it, fading toward the top
    float ex = (x - cx) / (0.45f * H), ey = (y - cy) / (0.55f * H);
    return sum * expf(-(ex * ex + ey * ey)) * smoothstepf(0.f, 0.35f * H, y);
}

__device__ __forceinline__ float luminance(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// final image: the figure (already in display colours) laid over everything behind it.
// The behind-light (universe, halo, aureole, glows, rays) goes through the film curve,
// then the two are combined with a "screen" blend: light adds to light, never past white.
__global__ void toPixels(const float3* light, const float3* tight, const float3* wide,
                         const float3* rays, const float3* sky, uchar3* out, int W, int H,
                         Mix m, Backdrop bd, uint32_t frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    float3 L = light[i], T = tight[i], G = wide[i], R = rays[i];

    // --- the figure, with a little shimmer from the tight glow, plus the jewel on its brow
    float jw = bd.jewel > 0.f ? bd.jewel * jewelGlow(x, y, bd.nimbusX, bd.jewelY, H) : 0.f;
    float3 fig = make_float3(fminf(L.x + m.tight * T.x + jw, 1.f),
                             fminf(L.y + m.tight * T.y + jw, 1.f),
                             fminf(L.z + m.tight * T.z + jw, 1.f));

    // --- what's behind the figure: the universe, nimbus, aureole, motes, aura, rays
    float3 back = make_float3(bd.skyLevel * sky[i].x, bd.skyLevel * sky[i].y, bd.skyLevel * sky[i].z);
    float nr = sqrtf((x - bd.nimbusX) * (x - bd.nimbusX) + (y - bd.nimbusY) * (y - bd.nimbusY));
    float sacred = bd.nimbusStrength * (expf(-powf((nr - bd.nimbusR) / bd.nimbusWidth, 2.f))   // the ring
                                        + 0.25f * expf(-powf(nr / bd.nimbusR, 2.f)));           // soft inner light
    if (bd.sunburst > 0.f) sacred += bd.sunburst * aureole(x, y, bd, H);
    if (bd.motes > 0.f)    sacred += bd.motes * risingMotes(x, y, bd.nimbusX, 0.5f * H, W, H, bd.t);
    float hide = 1.f / (1.f + bd.occlusion * luminance(T));          // the figure hides what's behind
    float3 behind = make_float3(back.x + m.exposure * (sacred * bd.nimbusColor.x + m.wide * G.x + m.rays * R.x) * hide,
                                back.y + m.exposure * (sacred * bd.nimbusColor.y + m.wide * G.y + m.rays * R.y) * hide,
                                back.z + m.exposure * (sacred * bd.nimbusColor.z + m.wide * G.z + m.rays * R.z) * hide);
    float3 bk = make_float3(powf(filmic(behind.x), 1.f / 2.2f), powf(filmic(behind.y), 1.f / 2.2f),
                            powf(filmic(behind.z), 1.f / 2.2f));

    float r = 1.f - (1.f - fig.x) * (1.f - bk.x);
    float g = 1.f - (1.f - fig.y) * (1.f - bk.y);
    float b = 1.f - (1.f - fig.z) * (1.f - bk.z);
    float dither = (hash(i * 9781u + frame * 6271u) * 2.3283064e-10f - 0.5f) / 255.f;  // kills banding
    auto to8 = [&](float v) { return (unsigned char)fminf(fmaxf((v + dither) * 255.f + 0.5f, 0.f), 255.f); };
    out[i] = make_uchar3(to8(r), to8(g), to8(b));
}

// ---- host helpers ---------------------------------------------------------------------
static std::vector<float> gaussian(float sigma, int& radius) {
    radius = (int)ceilf(3.f * sigma);
    std::vector<float> w(radius + 1);
    float sum = 0.f;
    for (int k = 0; k <= radius; ++k) { w[k] = expf(-0.5f * k * k / (sigma * sigma)); sum += (k ? 2.f : 1.f) * w[k]; }
    for (auto& v : w) v /= sum;
    return w;
}

// "How bright is bright?" — the p-th percentile of non-empty pixels (sampled every 7th px).
static float percentile(const uint32_t* h, size_t N, double p) {
    std::vector<uint32_t> v;
    for (size_t i = 0; i < N; i += 7) if (h[i]) v.push_back(h[i]);
    if (v.empty()) return 1.f;
    size_t k = std::min(v.size() - 1, (size_t)(p * v.size()));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return (float)std::max<uint32_t>(v[k], 1u);
}

// The choreography, as a function of loop time t in [0, 1): one smooth wave 0° -> 90° -> 0°.
static float angleAt(float t) {
    const float PI = 3.14159265f;
    return 0.5f * PI * 0.5f * (1.f - cosf(2.f * PI * t));
}

int main(int argc, char** argv) {
    const int W      = argc > 1 ? atoi(argv[1]) : 1920;
    const int H      = argc > 2 ? atoi(argv[2]) : 1080;
    const int frames = argc > 3 ? atoi(argv[3]) : 600;
    const int fStart = argc > 4 ? atoi(argv[4]) : 0;
    const int fEnd   = argc > 5 ? atoi(argv[5]) : frames;
    const int log2s  = argc > 6 ? atoi(argv[6]) : 28;   // 2^28 ≈ 268M random c's

    // ---- The look --------------------------------------------------------------------
    const float3 BAND_WEIGHT = make_float3(0.35f / 1.35f, 0.5f / 1.35f, 0.5f / 1.35f);  // fast, medium, slow
    const float3 RAY_TINT = make_float3(0.70f, 0.82f, 1.00f);   // pale moonbeams
    const float  RAY_THRESHOLD = 0.3f;                           // only bright light casts rays
    const Mix    MIX = { /*exposure*/ 0.25f, /*tight*/ 0.25f, /*wide*/ 0.35f, /*rays*/ 2.0f };
    const float  TARGET_DENSITY = 0.23f;                         // auto-exposure aims for this
    const float  BREATHS_PER_LOOP  = 3.f, BREATH_DEPTH = 0.18f;  // ±18% glow, 3 slow breaths
    Backdrop BD;
    BD.occlusion      = 8.0f;
    BD.nimbusColor    = make_float3(0.70f, 0.82f, 1.00f);
    BD.nimbusWidth    = 0.007f * H;
    BD.nimbusStrength = 1.0f;
    BD.skyLevel       = 0.55f;                                    // a dark sky lets the figure glow
    BD.sunburst       = 0.15f;
    BD.jewel          = 0.9f;
    BD.motes          = 3.0f;
    const float HEAD_Z = -1.0f, NIMBUS_RADIUS = 0.42f;            // in fractal units
    const float JEWEL_OFFSET = -0.07f;                             // third eye, relative to HEAD_Z

    const size_t N = (size_t)W * H;
    const dim3 tile(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    const float PI = 3.14159265f;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

    // ---- ONCE: find and sort the paths ---------------------------------------------------
    auto t0 = now();
    const uint64_t samples = 1ull << log2s, batch = 1ull << 24;
    const size_t capacity = samples / 2;   // in practice far fewer are kept; checked below
    float2* dC; int* dN; unsigned* dCount; unsigned count = 0;
    CK(cudaMalloc(&dC, capacity * sizeof(float2)));
    CK(cudaMalloc(&dN, capacity * sizeof(int)));
    CK(cudaMalloc(&dCount, sizeof(unsigned)));
    CK(cudaMemset(dCount, 0, sizeof(unsigned)));
    for (uint64_t first = 0; first < samples; first += batch)
        findPaths<<<(unsigned)(batch / 256), 256>>>(first, dC, dN, dCount);
    CK(cudaGetLastError());
    CK(cudaMemcpy(&count, dCount, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (count > capacity) { fprintf(stderr, "path buffer overflow\n"); return 1; }
    // longest paths first: lockstep groups get similar work, and the long ones start early
    thrust::sort_by_key(thrust::device, dN, dN + count, dC, thrust::greater<int>());
    CK(cudaDeviceSynchronize());
    fprintf(stderr, "kept %u of %llu random c's (%.1f%%) in %.2fs\n",
            count, (unsigned long long)samples, 100.0 * count / samples, secs(t0, now()));

    // ---- per-frame buffers ---------------------------------------------------------------
    uint32_t* dHist; float* dWarm; uchar3* dOut;
    float3 *dLight, *dTmp, *dTight, *dWide, *dRays;
    CK(cudaMalloc(&dHist, NBANDS * N * sizeof(uint32_t)));
    CK(cudaMalloc(&dWarm, N * sizeof(float)));
    float3* dSky;
    for (float3** p : {&dLight, &dTmp, &dTight, &dWide, &dRays, &dSky}) CK(cudaMalloc(p, N * sizeof(float3)));
    CK(cudaMalloc(&dOut, N * sizeof(uchar3)));

    // two blur sizes: tight (H/250 ≈ 4px at 1080p) and wide (H/25 ≈ 43px)
    auto upload = [](const std::vector<float>& v) {
        float* d; CK(cudaMalloc(&d, v.size() * sizeof(float)));
        CK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice)); return d; };
    int rTight, rWide;
    float* dTightW = upload(gaussian(H / 250.f, rTight));
    float* dWideW  = upload(gaussian(H / 25.f,  rWide));
    auto blur = [&](const float3* src, float3* dst, const float* w, int r) {
        blur1D<<<grid, tile>>>(src, dTmp, W, H, true,  w, r);
        blur1D<<<grid, tile>>>(dTmp, dst, W, H, false, w, r);
    };

    std::vector<uint32_t> hHist(NBANDS * N);
    std::vector<uchar3> hOut(N);
    float norm[NBANDS] = {}, autoExp = 1.f;

    auto tFrames = now();
    for (int f = fStart; f < fEnd; ++f) {
        float t = (float)f / frames;
        float theta = angleAt(t);
        float s2 = sinf(theta) * sinf(theta);   // 0 = Buddha view, 1 = Mandelbrot view

        Camera cam;
        cam.W = W; cam.H = H;
        cam.cosT = cosf(theta); cam.sinT = sinf(theta);
        cam.center = -0.20f - 0.55f * s2;          // Buddha is centred near -0.2, Mandelbrot near -0.75
        cam.zoom   = H / (3.00f - 0.60f * s2);     // Mandelbrot is smaller: ease in a little

        // 1. draw every kept path
        CK(cudaMemset(dHist, 0, NBANDS * N * sizeof(uint32_t)));
        drawPaths<<<(count + 255) / 256, 256>>>(cam, dC, dN, count, dHist);
        CK(cudaGetLastError());

        // 2. "full brightness" per band, and auto-exposure (both smoothed: no flicker)
        CK(cudaMemcpy(hHist.data(), dHist, NBANDS * N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int b = 0; b < NBANDS; ++b) {
            float p = percentile(hHist.data() + b * N, N, 0.999);
            norm[b] = (f == fStart) ? p : 0.8f * norm[b] + 0.2f * p;
        }
        const float wB[NBANDS] = {BAND_WEIGHT.x, BAND_WEIGHT.y, BAND_WEIGHT.z};
        double dSum = 0; size_t lit = 0;
        for (size_t i = 0; i < N; i += 7) {
            float v = 0.f;
            for (int b = 0; b < NBANDS; ++b) v += wB[b] * hHist[b * N + i] / norm[b];
            if (v > 0.01f) { dSum += v; ++lit; }
        }
        float meanD = (float)(dSum / std::max<size_t>(lit, 1));
        if (getenv("SHOW_DENSITY")) fprintf(stderr, " meanD=%.4f ", meanD);
        float target = TARGET_DENSITY > 0.f ? powf(TARGET_DENSITY / meanD, 0.7f) : 1.f;
        autoExp = (f == fStart) ? target : 0.8f * autoExp + 0.2f * target;

        // 3. counts -> coloured light -> glows + rays -> 8-bit pixels
        float breath = 1.f + BREATH_DEPTH * sinf(2.f * PI * BREATHS_PER_LOOP * t);
        Mix mix = MIX;
        mix.wide *= breath; mix.rays *= breath;
        Backdrop bd = BD;
        bd.nimbusX = 0.5f * W;
        bd.nimbusY = (HEAD_Z * (cam.cosT + cam.sinT) - cam.center) * cam.zoom + 0.5f * H;
        float rayY = bd.nimbusY + s2 * (0.62f * H - bd.nimbusY);  // behind the head -> centre of the Mandelbrot
        bd.nimbusR = NIMBUS_RADIUS * cam.zoom;
        float buddhaOnly = (1.f - s2) * (1.f - s2);             // sacred light: only while we see the Buddha
        bd.nimbusStrength *= breath * buddhaOnly;
        bd.sunburst *= breath * buddhaOnly;
        bd.motes    *= buddhaOnly;
        bd.jewel    *= (0.8f + 0.2f * breath / (1.f + BREATH_DEPTH)) * buddhaOnly;
        bd.jewelY = bd.nimbusY + JEWEL_OFFSET * cam.zoom;
        bd.t = t;

        countsToLight<<<(unsigned)((N + 255) / 256), 256>>>(dHist, dLight, dWarm, (int)N,
            make_float3(1.f / norm[0], 1.f / norm[1], 1.f / norm[2]), BAND_WEIGHT, autoExp);
        blur(dLight, dTight, dTightW, rTight);
        blur(dLight, dWide,  dWideW,  rWide);
        float raySource = (1.6f * NIMBUS_RADIUS + 3.f * s2) * cam.zoom;  // the halo -> the whole figure
        lightRays<<<grid, tile>>>(dWarm, dRays, W, H, 0.5f * W, rayY, raySource, RAY_THRESHOLD, RAY_TINT, (uint32_t)f);
        blur(dRays, dRays, dTightW, rTight);      // soften the ray texture a touch
        universe<<<grid, tile>>>(dSky, W, H, t, 0.5f * W, 0.45f * H);
        toPixels<<<grid, tile>>>(dLight, dTight, dWide, dRays, dSky, dOut, W, H, mix, bd, (uint32_t)f);
        CK(cudaGetLastError());

        CK(cudaMemcpy(hOut.data(), dOut, N * sizeof(uchar3), cudaMemcpyDeviceToHost));
        fwrite(hOut.data(), 1, N * 3, stdout);
        fflush(stdout);
        int done = f - fStart + 1;
        fprintf(stderr, "\rframe %d/%d  theta=%5.1f deg  %.1f fps", f + 1, fEnd, theta * 180.f / PI,
                done / secs(tFrames, now()));
    }
    fprintf(stderr, "\n");
    return 0;
}
