Interrupted development attempt. Source snapshots and saved encounters are retained.

The context question guard mistakenly rejected the existing canonical severity
question containing 'say'. This made canonical extraction all-uncertain. In
addition, the one-field prompt often selected missing or uncertainty for explicit
facts. No inference from this incomplete attempt is a confirmatory result.
Next attempt restores exact legacy question compatibility and uses the existing
five-slot few-shot prompt with assistant-prefix constrained continuation.
