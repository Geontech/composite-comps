Tested pipeline configurations. These ship in the fleet image at
`/usr/share/composite-comps/examples/`.

`all-modules-load.json` instantiates every module the fleet builds, with no connections. It is the
graph `build:container` runs as its smoke gate: it proves each module loads, configures, and shuts
down cleanly inside the image, rather than only that it can be dlopen'd.
