import unittest
import torch
from transport_reference import (LoopLayout, loop_mix, previous_token_states,
                                 fixed_feedback, inject_previous_top)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.s = torch.randn(2, 7, 6, 5, dtype=torch.float64)
        self.top = torch.randn(2, 7, 5, dtype=torch.float64)
        self.seg = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2]])
        self.valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1]], dtype=torch.bool)
        self.bos = torch.zeros_like(self.valid); self.bos[1, 5] = True
        self.ids = ('a', 'b')

    def shift(self, s=None, top=None, sl=slice(None), carry=None, ids=None):
        return previous_token_states((self.s if s is None else s)[:, sl],
            (self.top if top is None else top)[:, sl], self.seg[:, sl],
            self.valid[:, sl], self.bos[:, sl], self.ids if ids is None else ids, carry)

    def test_cache_slots_unique_contiguous(self):
        for steps in (1, 2, 3):
            c = LoopLayout(8, 2, 6, steps, 2, 1)
            slots = [c.cache_slot(i) for i in (0, 1, 6, 7)]
            slots += [c.cache_slot(i, s) for s in range(steps) for i in range(2, 6)]
            self.assertEqual(sorted(slots), list(range(c.cache_layers)))

    def test_partial_and_window_sources(self):
        c = LoopLayout(8, 2, 6, 4, 2, 1)
        self.assertEqual(c.sources(2, 2, 'attn'), (('prelude', -1, -1), ('block', 1, 0), ('block', 1, 1)))
        self.assertEqual(c.sources(2, 2, 'mlp')[-1], ('partial', 2, 0))
        self.assertEqual(c.sources(2, 3, 'attn')[-1], ('partial', 2, 0))
        self.assertEqual(c.sources(2, 4, 'attn')[-1], ('block', 2, 0))
        self.assertEqual(c.readout_sources()[-1], ('block', 3, 1))
        self.assertEqual(len(LoopLayout(8, 2, 6, 3, 2, 0).readout_sources()), 3)

    def test_mix_matches_explicit_reference(self):
        v = torch.randn(3, 2, 4, 5, dtype=torch.float64, requires_grad=True)
        k, q, z, g = [torch.randn(*shape, dtype=v.dtype) for shape in [(3, 5), (5,), (5,), (5,)]]
        expected = torch.zeros_like(v[0])
        for b in range(2):
            for t in range(4):
                logits = torch.stack([k[i] @ q + ((v[i,b,t] * g) @ z) / (v[i,b,t].square().mean()+1e-6).sqrt() for i in range(3)])
                expected[b,t] = sum(a * w for a, w in zip(v[:,b,t], logits.softmax(0)))
        out = loop_mix(v, k, q, z, g)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(torch.autograd.grad(out.square().sum(), v)[0],
                                   torch.autograd.grad(expected.square().sum(), v)[0])

    def test_mix_no_cross_token_or_row(self):
        v = torch.randn(3, 2, 4, 5); changed = v.clone(); changed[:, 1, 3] += 200
        k = torch.randn(3,5); q,z,g = (torch.randn(5) for _ in range(3))
        a,b = (loop_mix(x,k,q,z,g) for x in (v,changed))
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
        torch.testing.assert_close(a[1,:3],b[1,:3],rtol=0,atol=0)

    def test_shift_mask(self):
        p, top, mask, _ = self.shift()
        self.assertEqual(mask.tolist(), [[False,True,True,False,True,True,True], [False,True,False,False,True,False,True]])
        torch.testing.assert_close(p[0,1],self.s[0,0]); torch.testing.assert_close(top[0,4],self.top[0,3])
        self.assertTrue(torch.equal(p[~mask],torch.zeros_like(p[~mask])))
        self.assertTrue(torch.equal(top[~mask],torch.zeros_like(top[~mask])))

    def test_chunk_equivalence_all_splits(self):
        whole = self.shift()
        for split in range(1,7):
            left = self.shift(sl=slice(0,split)); right = self.shift(sl=slice(split,None),carry=left[3])
            for i in range(3):
                torch.testing.assert_close(torch.cat((left[i],right[i]),dim=1),whole[i],rtol=0,atol=0)

    def test_reordered_carry_fails(self):
        carry = self.shift(sl=slice(0,2))[3]
        with self.assertRaisesRegex(ValueError,'reordered'):
            self.shift(sl=slice(2,None),carry=carry,ids=('b','a'))

    def test_shift_causality_and_isolation(self):
        changed = self.s.clone(); changed[0,2] += 123; changed[1,6] += 500
        a,b = self.shift()[0],self.shift(s=changed)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)  # 0,2 is followed by segment reset; final token has no successor
        changed[0,0] += 9
        b = self.shift(s=changed)[0]
        self.assertFalse(torch.equal(a[0,1],b[0,1]))
        torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
        torch.testing.assert_close(a[0,2:],b[0,2:],rtol=0,atol=0)

    def test_fixed_window_and_mask(self):
        p,_,mask,_ = self.shift(); x = torch.randn_like(self.top)
        changed = p.clone(); changed[:,:,:2] += 200; changed[:,:,4:] -= 100
        out = fixed_feedback(x,p,mask,2,2,0.)
        torch.testing.assert_close(out,fixed_feedback(x,changed,mask,2,2,0.),rtol=0,atol=0)
        torch.testing.assert_close(out[~mask],x[~mask],rtol=0,atol=0)
        m = p[:,:,2:4].mean(2); scale = ((x.square().mean(-1,keepdim=True)+1e-6)/(m.square().mean(-1,keepdim=True)+1e-6)).sqrt()
        torch.testing.assert_close(out[mask],(.5*x+.5*m*scale)[mask])

    def test_top_injection_zero_and_mask(self):
        _,p,mask,_ = self.shift(); x = torch.randn_like(self.top)
        ident = torch.eye(5,dtype=x.dtype)
        torch.testing.assert_close(inject_previous_top(x,p,mask,ident*0),x,rtol=0,atol=0)
        out=inject_previous_top(x,p,mask,ident)
        torch.testing.assert_close(out[~mask],x[~mask],rtol=0,atol=0)
        self.assertFalse(torch.equal(out[mask],x[mask]))

    def test_feedback_gradients_finite(self):
        p,top,mask,_=self.shift(); x=torch.randn_like(self.top,requires_grad=True)
        p.requires_grad_(); top.requires_grad_(); logits=torch.tensor(0.,dtype=x.dtype,requires_grad=True)
        proj=torch.eye(5,dtype=x.dtype,requires_grad=True)
        y=fixed_feedback(x,p,mask,3,2,logits)+inject_previous_top(x,top,mask,proj)
        y.square().sum().backward()
        for tensor in (x,p,top,logits,proj):
            self.assertTrue(torch.isfinite(tensor.grad).all()); self.assertGreater(tensor.grad.abs().sum().item(),0.)
        self.assertEqual(p.grad[~mask].abs().sum().item(),0.)
        self.assertEqual(top.grad[~mask].abs().sum().item(),0.)

    def test_reject_invalid_layout_and_metadata(self):
        with self.assertRaises(ValueError): LoopLayout(8,2,6,3,3,1)
        with self.assertRaises(ValueError): LoopLayout(8,2,6,3,2,1).cache_slot(0,1)
        with self.assertRaises(ValueError): previous_token_states(self.s[:,:,:5],self.top,self.seg,self.valid,self.bos,self.ids)
        with self.assertRaises(ValueError): fixed_feedback(self.top,self.s,self.valid,6,1,0.)


if __name__ == '__main__': unittest.main()
