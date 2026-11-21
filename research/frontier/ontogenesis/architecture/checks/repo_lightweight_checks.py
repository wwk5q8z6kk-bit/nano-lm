"""Pure-stdlib sanity checks for Nano LM theory propositions.
These are deterministic examples, not formal proof or LM benchmarks.
"""
import itertools
import json
import math


def check_versioned():
    events = [('item','old',1,'a'), ('item','new',2,'b'), ('item',None,3,'c')]
    def get(es, time):
        eligible = [e for e in es if e[0] == 'item' and e[2] <= time]
        if not eligible:
            return None
        return max(eligible,key=lambda e:(e[2],e[3]))[1]
    assert all(get(p,1)=='old' and get(p,2)=='new' and get(p,3) is None
               for p in itertools.permutations(events))


def check_transport():
    z = (2.0, 5.0)
    w = (3.0, 7.0)
    original = sum(a*b for a,b in zip(w,z))
    transformed_z = (z[1],z[0])
    transformed_w = (w[1],w[0])
    assert abs(original - sum(a*b for a,b in zip(transformed_w,transformed_z))) < 1e-12


def check_first_order_nullspace():
    # J=[1,0], so a projected update must have first coordinate exactly 0.
    jac = (1.0,0.0)
    new_gradient = (2.0,3.0)
    projected = (0.0,new_gradient[1])
    assert sum(a*b for a,b in zip(jac,projected)) == 0.0
    # Conflict in the constrained direction is removed, demonstrating the trade-off.
    assert (new_gradient[0],0.0) != projected


def check_storage():
    n, v = 4, 8
    b = math.ceil(n*math.log2(v))
    assert b==12 and 2**(b-1) < v**n <= 2**b


def check_compute_gate():
    threshold = 0.10 + 0.02
    assert 0.14 > threshold and not 0.09 > threshold



def check_bounded_logprob():
    a=[2.,0.,-1.,1.]; b=[2.04,-.04,-.97,.98]
    eps=max(abs(x-y) for x,y in zip(a,b))
    def logp(z):
        maxz=max(z)
        norm=maxz+math.log(sum(math.exp(v-maxz) for v in z))
        return [v-norm for v in z]
    p,q=logp(a),logp(b)
    assert max(abs(x-y) for x,y in zip(p,q))<=2*eps+1e-12
    assert 0<=sum(math.exp(x)*(x-y) for x,y in zip(p,q))<=2*eps+1e-12

def check_nonzero_second_order_drift():
    h=.13
    # f(theta0,theta1)=theta0+theta1**2/2; J(0,0)=[1,0]
    assert 0<.5*h*h<=h*h/2+1e-14

def check_myopic_can_fail():
    assert -.2<0<-.2+1.

def check_bitemporal_visibility():
    ev=[("old",1,1,"a"),("corrected",1,5,"b")]
    def get(t,r):
        eligible=[e for e in ev if e[1]<=t and e[2]<=r]
        return max(eligible,key=lambda e:(e[1],e[2],e[3]))[0] if eligible else None
    assert get(2,2)=="old" and get(2,6)=="corrected"

def main():
    checks = [check_versioned,check_transport,check_first_order_nullspace,
              check_storage,check_compute_gate,check_bounded_logprob,
              check_nonzero_second_order_drift,check_myopic_can_fail,
              check_bitemporal_visibility]
    passed = []
    for fn in checks:
        fn()
        passed.append(fn.__name__)
    print(json.dumps({'passed':len(passed),'total':len(checks),'checks':passed,
                      'scope':'Deterministic mathematical examples; no model training or superiority claim'},indent=2))


if __name__ == '__main__':
    main()
