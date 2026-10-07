import sys
import os

# Add the parent directory to sys.path to allow importing nanoscribe modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoscribe.campaign import estimate_pod_cost

def test_estimate_pod_cost_happy_path():
    assert estimate_pod_cost(1.5, 2.0) == 3.0
    assert estimate_pod_cost(2.0, 3.5) == 7.0

def test_estimate_pod_cost_zero():
    assert estimate_pod_cost(0.0, 5.0) == 0.0
    assert estimate_pod_cost(1.5, 0.0) == 0.0
    assert estimate_pod_cost(0.0, 0.0) == 0.0
