# X07 byte accounting correction

The original report and checksummed results use 128 bytes for the fast sketch. That figure measures the configured value capacity. It omits the retained subject identifiers, keys and polarities. The reported 1,810.7-byte combined figure inherits that omission. Both figures are historical estimates and do not measure process memory.

The corrected controller exposes the value capacity separately and includes UTF-8 identity and polarity payloads in its logical fast-state estimate. It truncates values on UTF-8 boundaries. Python object and allocator overhead remain outside this estimate. The historical experiment has not been rerun with the corrected implementation.
