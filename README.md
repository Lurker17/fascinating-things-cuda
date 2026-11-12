# fascinating-things-cuda

Small, self-contained CUDA programs that turn a bit of mathematics into something beautiful to watch.
Each one is a single `.cu` file, commented so that it can be read top to bottom and understood.

| Project | What it is |
|---|---|
| [buddhabrot](buddhabrot/) | The escape paths of the Mandelbrot iteration form a seated Buddha. A 4D camera turns it into the Mandelbrot set and back, in a seamless 20-second loop. |

[![Buddhabrot → Mandelbrot](buddhabrot/preview.jpg)](buddhabrot/)

## Requirements

- An NVIDIA GPU and the CUDA toolkit (`nvcc`)
- `ffmpeg`, to turn the raw frames into a video
