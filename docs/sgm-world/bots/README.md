# SGM Data World worker bots

These are the little orange bots that run along the roads in SGM Data World and carry data from real buildings to the laptop in Bengaluru. They are blocky terracotta characters in the Claude Code mascot style: a wide orange block with two dark rectangular eyes, stub arms and short legs. Each one wears a T-shirt that is the flag of the country its data comes from, wrapped all the way round. There is no logo or text on any of them.

`build_bots.py` writes all ten models. It uses only the Python 3 standard library and writes binary glTF (`.glb`) directly, with the flag textures drawn in code and saved as embedded PNGs. `build_props.py` makes the ships, submarine, ML bot, toy figure, laptop, rupee notes and data crate (see [Props](#props) below). Running it twice gives byte-identical files.

## Regenerate

```sh
python3 docs/sgm-world/bots/build_bots.py                   # build all 10 models, then validate them
python3 docs/sgm-world/bots/build_bots.py --check           # only validate the files already on disk
python3 docs/sgm-world/bots/build_bots.py --dump-textures /tmp/tees   # also write the tee PNGs
```

The models go to `shreeganeshmodel-deploy/assets/bots/`. The validator reads each file back and checks the GLB header and chunk lengths, that every bufferView and accessor fits in the buffer, that indices stay in range, that `run` and `idle` exist and loop cleanly (first key equals last key), that both clips animate the same nodes, that the feet sit on y = 0, and that each file stays under 150 KB and 3,000 triangles.

To look at them, serve the deploy folder and open the preview page:

```sh
cd shreeganeshmodel-deploy && python3 -m http.server 8794 --bind 127.0.0.1
# http://localhost:8794/assets/bots/preview.html
```

The page has buttons for run, idle, pause, turntable and backs. URL options for screenshots: `?anim=idle`, `?t=0.11` (freeze at that clip time), `?only=bot_IN,bot_US`, `?back=1`, `?labels=0`, and `?focus=bot_IN` for a close-up (adjust with `fy`, `fd`, `fx`, or `cam=x,y,z`). `preview_row.png`, `preview_IN_closeup.png` and `preview_IN_back.png` in this folder were captured from it.

## Files

| File | What it is |
|---|---|
| `bot_IN.glb`, `bot_US.glb`, `bot_DE.glb`, `bot_AE.glb` | Country bots, hands free |
| `bot_astro.glb` | Astronaut bot for future satellite data: white suit, round glass helmet, small jetpack, no flag |
| `bot_*_carry.glb` (5 files) | Same bots holding a glowing data crate above the head with both arms up |

Sizes are about 58–87 KB for the country bots and 100–128 KB for the astro bots. Triangle counts are 512 (country), 812 (country carry), 1,604 (astro) and 2,080 (astro carry).

## Conventions

- glTF 2.0, +Y up, the bot faces +Z, so its left hand is on +X.
- The origin is on the ground, centred between the feet. Rest-pose height is 0.98 for the country bots, 1.34 with the crate, 1.24 for the astro (helmet) and 1.59 for the astro with the crate. The units have no fixed scale; the app scales the models.
- Materials are PBR metallic-roughness with metallic 0 and roughness 0.8, except the eyes (0.45), the crate (0.4 core and 0.6 frame) and the helmet glass (0.06, alpha-blended at 0.24). The crate core and the jet flames are emissive.
- The tee texture is a 2048×256 panorama: u runs round the body from the front-left edge to the viewer's right, and v runs down the tee. It is sampled linear with mipmaps and clamped at the edges.
- Animations work only through node TRS channels with linear interpolation. There is no skinning and no glTF extension.

| Clip | Length | What moves |
|---|---|---|
| `run` | 0.45 s, one full stride (two footfalls) | legs swing ±36° and the swinging leg lifts; the body bounces 0.05, lowest at mid-stance; the torso leans forward 9° ±2.5° with a small roll and twist; squash on contact and stretch in the air; arms swing ±55° opposite the legs (carry bots keep their arms up); astro jet flames flicker |
| `idle` | 2.0 s | breathing (torso scale ±2%), a slight sway, arms drift out and back, one blink at 1.3 s, astro flames low |

Both clips animate the same set of nodes, so switching between them never leaves a part in an old pose.

Node names: `bot_XX` (root) → `hips` → `leg_L`, `leg_R`, `torso` → `eyes`, `arm_L`, `arm_R`, plus `crate` on carry bots and `helmet`, `jetpack` → `flame_L`, `flame_R` on astro bots.

## Flag rules

The flag goes only on the T-shirt, on the upper body. It never goes on pants, shoes, hats or anything below the waist, because India's flag code forbids that, and the same rule is used for every country.

The whole tee is the flag, wrapped all the way round, so it reads from every side. The bands run horizontally round the body, with flag height equal to the visible tee height. The emblem sits on the front and is repeated on the back, with the hoist on the viewer's left from either side: the Ashoka Chakra is centred, the US starred canton is at the upper left, and the UAE red band is on the left. The sloped shoulders take the top band colour. The sleeves are one plain matching flag colour: IN saffron, US canton blue, DE black and AE green.

| Code | Flag | Ratio | How it is drawn |
|---|---|---|---|
| IN | India | 2:3 | equal saffron #FF9933 / white / India green #138808 bands; navy #000080 Ashoka Chakra (24 spokes, rim, hub, 24 rim dots) centred in the white band, front and back |
| US | United States | 10:19 | 13 stripes with red #B22234 at top and bottom, all round; blue #3C3B6E canton 7 stripes tall and 0.76 of the height wide at the upper left of the front and of the back; all 50 white stars in 9 rows that alternate 6 and 5 |
| DE | Germany | 3:5 | black / red #DD0000 / gold #FFCE00, all round |
| AE | United Arab Emirates | 1:2 | green #00732F / white / black bands all round; red #FF0000 vertical band (half the flag height wide, which is a quarter of a 1:2 flag's length) at the left of the front and of the back |

Pants are plain colours: IN navy, US denim, DE charcoal, AE sand. Shoes are dark with a light sole. The astro has white suit legs and grey boots.

## Known limitations

- Stub arms can't reach above the big head. To carry the crate, the carry models use longer, stretched arms with the hands hooked over the crate's lower sides, and the crate rests just on top of the head. On the astro, the arms have to clear the round helmet, so they bend outward and then up like a goal post. This works at road scale but looks a little gangly in close-up.
- Flat shading and 45° bevels give a low-poly toy look on purpose. Nothing has been tuned yet for Cesium's lighting, so check the colours in the world once the bots are added.

## Props

`build_props.py` imports the mesh, texture and glTF code from `build_bots.py` and writes eight props to `shreeganeshmodel-deploy/assets/props/`. It follows the same conventions: +Y up, faces +Z, origin at the centre of the base, node-TRS animations only, deterministic output. Each prop must stay under 150 KB and 5,000 triangles.

```sh
python3 docs/sgm-world/bots/build_props.py            # build all props, then validate them
python3 docs/sgm-world/bots/build_props.py --check    # only validate
# preview: http://localhost:8794/assets/props/props_preview.html (each prop is scaled to one slot; the label gives its real size)
```

| File | What it is | Size (units) | Clips |
|---|---|---|---|
| `tanker_oil.glb` | Crude oil tanker (Ship 1, oil). It has a black upper hull over a red-brown bottom band and a green deck. On deck are the centreline pipe run, a cross manifold with valves midships, three cranes, a raised catwalk, and a forecastle with a foremast. At the stern sit a white accommodation block with bridge wings and windows, a funnel and a free-fall lifeboat. The raked bow has a bulbous-bow hint. | 2.0 × 3.5 × 12.0 | `idle` 4.0 s (bob, slow roll) |
| `container_ship.glb` | Container ship (Ship 2): blue hull with a red bottom band, bridge block three-quarters aft, mixed-colour container stacks forward and aft of it, funnel aft, breakwater and bulbous bow | 2.1 × 4.3 × 12.0 | `idle` 3.0 s (bob) |
| `submarine.glb` | Yellow-orange cartoon submarine: tower with periscope and sail planes, portholes, cruciform tail fins, four-blade propeller (`propeller` node) | 2.2 × 3.0 × 8.3 | `idle` 1.5 s (propeller spin, slow bob) |
| `bot_ML.glb` | ML bot: the worker-bot body with a dark helmet and two pointy ears, a dark tee with a small white neural-net pattern (no flag), holding a hammer | 0.9 × 1.24 × 0.7 | `idle` 2.0 s, `hammer` 0.6 s |
| `me_figure.glb` | Toy figure of the site owner: curly black hair, round glasses, medium-brown skin, smile, white tee printed "ME", jeans, sneakers. The `hair`, `glasses`, `face` and `tshirt` nodes are separate so they can be restyled from a photo. | 0.6 × 1.0 × 0.45 | `idle` 2.4 s, `wave` 1.2 s (right arm) |
| `laptop_giant.glb` | Open laptop at about 105°: silver body, dark keys, and a screen showing a rising stock line labelled "STOCKS" as an emissive texture. A soft halo plane (`glow` node) pulses. | 1.7 × 1.1 × 1.4 | `idle` 2.0 s (glow pulse) |
| `rupee_note.glb` | Bundle of generic rupee notes (lavender and pale green, with a ₹ sign and "500"; not a real banknote design and no portrait) with a paper band, plus a gold ₹ coin | 1.35 × 0.3 × 0.5 | `idle` 2.0 s (one full turn) |
| `data_crate.glb` | The glowing data crate from the carry bots, scaled to 0.5 wide | 0.5 × 0.4 × 0.43 | `idle` 1.6 s (float) |

Notes:
- glTF can't animate emissive strength with TRS channels, so the laptop's glow pulse is a scale pulse on a soft, alpha-blended halo plane in front of the screen.
- Ships and the submarine bob around a pivot node (`hull` or `sub`) near the waterline. Their rest pose sits on y = 0.
- `ship_cargo.glb` from the first request was dropped in favour of the tanker and the container ship.
