# Two posts across a guide drawn along y in SolidCraft

`two-posts.sc` was drawn in SolidCraft 0.42 by hand (the README's demo recording): a
22.86 × 10.16 mm rectangle sketched on the XZ plane and extruded 60 mm along −y as the body
`guide`, and two full-height cylinders of 3 mm diameter, `post1` and `post2`, at y = −20 and
−40 mm on the guide's centre line, each a body of its own. It is SolidCraft's own document as
saved (its JSON, without the evaluated cache).

The case opens it with `guide` as the region (the posts stay metal, the default), puts the ports
by `end` alone, which for a guide running along y must take the two ends across y (not the broad
walls across z, `Model::long_end`), and solves it by the adaptive recipe of `design-post`. The
two posts form a resonator (its peak near 9.74 GHz); the reference at 10 GHz, on the resonance's
flank, is emcraft-theory's `Posts`: the same Nyström method as the single post with each post's
field acting on the other, computed separately from the finite elements
(`cargo run -p emcraft-theory --example posts_reference -- 10`).
