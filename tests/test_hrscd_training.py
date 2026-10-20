import unittest
import numpy as np
import torch
from datasets.hrscd import transform_aligned
from utils.hrscd_training import Metrics, supervised_loss
from utils.hrscd_graph import audit_regions
from utils.sam_node_audit import GT, IGNORE

class HRSCDTests(unittest.TestCase):
    def test_shared_geometry(self):
        a=np.arange(16).reshape(4,4)
        for k in range(4):
            images=transform_aligned([np.stack([a]*3,-1),a,a+1,a%2],k,True,False)
            np.testing.assert_array_equal(images[0][...,0],images[1])
            np.testing.assert_array_equal(images[2],images[1]+1)
            np.testing.assert_array_equal(images[3],images[1]%2)

    def test_independent_binary_target_and_unknown_semantics(self):
        a=torch.zeros(1,6,4,4,requires_grad=True); b=a.detach().clone().requires_grad_()
        c=torch.zeros(1,1,4,4,requires_grad=True)
        batch={'label_A':torch.zeros(1,4,4,dtype=torch.long),'label_B':torch.zeros(1,4,4,dtype=torch.long),
               'change':torch.ones(1,4,4,dtype=torch.long)}
        loss,_=supervised_loss((a,b,c),batch);loss.backward()
        self.assertEqual(float(a.grad.abs().sum()),0.)
        self.assertLess(float(c.grad.sum()),0.)

    def test_full_land_cover_is_not_change(self):
        a=torch.zeros(1,6,4,4,requires_grad=True);b=a.detach().clone().requires_grad_()
        c=torch.zeros(1,1,4,4,requires_grad=True)
        batch={'label_A':torch.ones(1,4,4,dtype=torch.long),'label_B':torch.ones(1,4,4,dtype=torch.long),
               'change':torch.zeros(1,4,4,dtype=torch.long)}
        loss,_=supervised_loss((a,b,c),batch);loss.backward()
        self.assertGreater(float(c.grad.sum()),0.)

    def test_stable_labeled_nodes_supervised_unknown_ignored(self):
        masks=torch.ones(1,4,4,dtype=torch.bool)
        a=torch.ones(4,4,dtype=torch.long);unknown=torch.zeros_like(a)
        audit=audit_regions(masks,a,unknown,unknown)
        self.assertEqual(audit.states_a.item(),GT)
        self.assertEqual(audit.states_b.item(),IGNORE)
        self.assertEqual(audit.change_ratio.item(),0.)

    def test_perfect_metrics(self):
        target=torch.tensor([[[1,2],[0,3]]]);change=torch.tensor([[[0,1],[0,1]]])
        logits=torch.nn.functional.one_hot(target,6).permute(0,3,1,2).float()*10
        batch={'label_A':target,'label_B':target,'change':change}
        metric=Metrics();metric.update((logits,logits,(change*20-10).unsqueeze(1)),batch)
        result=metric.result()
        self.assertEqual(result['Fscd'],1.)
        self.assertEqual(result['semantic_mIoU_present'],1.)
        self.assertEqual(result['change_F1'],1.)

if __name__=='__main__':unittest.main()
