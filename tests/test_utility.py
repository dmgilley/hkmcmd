#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import unittest
import numpy as np
from copy import deepcopy
from hybrid_mdmc.utility import *


class Testutility(unittest.TestCase):

    def setUp(self):
        return
    
    def test_isfloat_str(self):
        self.assertTrue(isfloat_str("test"))
        self.assertTrue(isfloat_str("1"))
        self.assertTrue(isfloat_str("1.0"))
        test = "test"
        self.assertTrue(isfloat_str(test))
        test = "1"
        self.assertTrue(isfloat_str(test))
        test = "1.0"
        self.assertTrue(isfloat_str(test))
        self.assertFalse(isfloat_str(1))
        self.assertFalse(isfloat_str(1.0))
        self.assertFalse(isfloat_str([]))
        return

    def test_unwrap_coordinates(self):
        coordinates = np.array([

            # unwrap on edge

            # unwrap, just greater than half box length

            # do not unwrap, equals half box length

            # unwrap three atoms

            # solo atom, do not unwrap

            # solo atom outside box, do not unwrap

            # unwrap only x

            # unwrap only y

            # unwrap only z

        ])
        adj_list = [
        ]
        box = [ [-10.0,10.0], [-8.0,8.0], [-6.0,6.0] ]
        return



if __name__ == "__main__":
    unittest.main()
