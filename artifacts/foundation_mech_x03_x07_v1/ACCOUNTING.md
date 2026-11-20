# X03–X07 measurement corrections

## X07 byte accounting

The original report and checksummed results use 128 bytes for the fast sketch. That figure measures the configured value capacity. It omits the retained subject identifiers, keys and polarities. The reported 1,810.7-byte combined figure inherits that omission. Both figures are historical estimates and do not measure process memory.

The corrected controller exposes the value capacity separately and includes UTF-8 identity and polarity payloads in its logical fast-state estimate. It truncates values on UTF-8 boundaries. Python object and allocator overhead remain outside this estimate. The historical experiment has not been rerun with the corrected implementation.


## X03–X05 mechanism comparisons

The archived GDN2 runs erased along the value dimension. This could erase a value held by an unrelated key. The corrected implementation erases along the key dimension used by its predictions. The archived differential GQA baseline applied a rotary transform across the two attention branches. The corrected baseline keeps each rotary coordinate pair inside its branch, preserving the relative-position property of each score.

The GDN2 and differential GQA scores in the original report describe their respective earlier implementations. Comparisons involving either corrected arm require fresh measurements. The regression suite checks preservation of an orthogonal key after channel decay, relative-position invariance within each attention branch, and sequence-versus-step parity.
