"""Inverse Fourier peak localization uses phase only, not known shift input."""
import unittest
import numpy as np
from xian_r5.alignment.subpixel_phase import refine_zero_peak


class SubpixelTests(unittest.TestCase):
    def test_fractional_peak(self):
        fy=np.fft.fftfreq(64)[:,None];fx=np.fft.fftfreq(64)[None,:]
        phase=np.exp(-2j*np.pi*(fy*(-.375)+fx*.25))
        self.assertEqual(refine_zero_peak(phase),(.25,-.375))

    def test_exact_zero_stays_exact_zero(self):
        self.assertEqual(refine_zero_peak(np.ones((64,64),complex)),(0.,0.))

    def test_dc_only_is_not_unique_position(self):
        phase=np.zeros((64,64),complex);phase[0,0]=1
        with self.assertRaises(ValueError):refine_zero_peak(phase)

    def test_unknowns_do_not_become_positions(self):
        phase=np.ones((64,64),complex);phase[0,0]=np.nan
        with self.assertRaises(ValueError):refine_zero_peak(phase)


if __name__=='__main__':unittest.main()
