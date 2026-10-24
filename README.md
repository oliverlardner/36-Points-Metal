# 36 Points Metal

A Metal port of [36 Points](https://www.sagejenson.com/36points/) (2019–2022) by Sage Jenson.

The systems, parameter sets, and steering are his. This project reimplements that simulation in Metal and plays it fullscreen, with the field drawn twice across the display.

## License

[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).

You can share and adapt this port for non-commercial use. Credit Sage Jenson, link the original and this license, say that it was adapted, and keep adaptations under the same license. Contact him for other licensing.

## Run

macOS 14 or later, and Xcode 16 or later. Open `36PointsMetal.xcodeproj` and run it. It opens fullscreen; Escape or Control-Command-F returns to the window.

The window title shows the current system. Left and right arrows move through all 43. `R` clears the field and starts that system again.

## Embed it

`ThirtySixPoints` renders into `displayTexture`, a square `rgba16Float` texture.

- `step()` advances one frame. Call it about 60 times a second. The fade and blur amounts were tuned for that rate.
- `select(_:)` blends into another system, using the same parameter mix as the original.
- `inverted` is true for `ink_on_white` and `strike`, which the original draws as black lines on white.
- `name` and `index` identify the current system. The full list is in `PointsCatalog`.
