"""Campaign orchestration helpers (manifests, revalidation waves, imports)."""

# Re-export the spend ledger. nanoscribe/campaign.py was converted into this
# package without carrying its contents across, so the module was shadowed and
# CampaignLedger / DEFAULT_LEDGER became unimportable — breaking every caller.
# The original module now lives at nanoscribe/campaign/ledger.py.
from nanoscribe.campaign.ledger import (  # noqa: E402,F401
    CampaignLedger,
    DEFAULT_LEDGER,
)

__all__ = ["CampaignLedger", "DEFAULT_LEDGER"]
