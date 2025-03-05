"""
test_SimplifiedThreePL.py

This file contains the unit and integration tests for the SimplifiedThreePL class.
- The Experiment.py and SignalDetection.py files are provided by the professor.
- The test code (and parts of the SimplifiedThreePL implementation) were generated with assistance from ChatGPT o3-mini.

Author: Aiden Hai
Date: 03/04/2025
"""


import sys
import os
# Insert the repository root (parent of src and tests) into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):
    
    def setUp(self):
        # Create an Experiment with 5 conditions using simple test data.
        # For each condition, we use a simple SignalDetection object.
        self.exp = Experiment()
        # For example, each condition has 50 trials with ~35 correct responses (example values).
        for _ in range(5):
            # For example: hits=20, misses=10, falseAlarms=5, correctRejections=15
            # n_correct = 20 + 15 = 35; n_incorrect = 10 + 5 = 15; total trials = 50.
            sdt = SignalDetection(20, 10, 5, 15)
            self.exp.add_condition(sdt)
        
        self.model = SimplifiedThreePL(self.exp)
    
    # Initialization Tests
    def test_valid_initialization(self):
        # Test that the constructor properly handles valid input.
        summary = self.model.summary()
        self.assertEqual(summary["n_conditions"], 5)
        self.assertEqual(summary["n_total"], int(np.sum(self.model._correct_array + self.model._incorrect_array)))
    
    def test_invalid_initialization(self):
        # Test that the constructor raises an error if the number of conditions is not 5.
        exp_invalid = Experiment()
        for _ in range(3):
            sdt = SignalDetection(10, 5, 3, 7)
            exp_invalid.add_condition(sdt)
        with self.assertRaises(ValueError):
            SimplifiedThreePL(exp_invalid)
    
    def test_get_params_before_fit(self):
        # Test that accessing parameters before calling fit() raises an error.
        with self.assertRaises(ValueError):
            self.model.get_discrimination()    # It's not fitted yet!
        with self.assertRaises(ValueError):
            self.model.get_base_rate()    # It's not fitted yet!
    
    # === Prediction Tests ===
    def test_predict_range(self):
        # Test that predict() returns values between 0 and 1.
        a = 1.0
        c = 0.2
        q = self.model._base_rate_to_logit(c)
        preds = self.model.predict((a, q))
        self.assertEqual(len(preds), 5)
        for p in preds:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
    
    def test_base_rate_effect(self):
        # Test that with all else equal, a higher base rate yields higher predicted probabilities.
        a = 1.0
        c_low = 0.2
        c_high = 0.4
        q_low = self.model._base_rate_to_logit(c_low)
        q_high = self.model._base_rate_to_logit(c_high)
        p_low = self.model.predict((a, q_low))
        p_high = self.model.predict((a, q_high))
        for p_l, p_h in zip(p_low, p_high):
            self.assertGreater(p_h, p_l)
    
    def test_difficulty_effect_positive_a(self):
        # For positive a, higher difficulty (larger b) should yield lower predicted probabilities.
        a = 1.0
        c = 0.3
        q = self.model._base_rate_to_logit(c)
        preds = self.model.predict((a, q))
        # With difficulties [2, 1, 0, -1, -2], we expect the condition with b=2 to have the lowest probability, and b=-2 to have the highest.
        self.assertLess(preds[0], preds[-1])
        self.assertLess(preds[0], preds[1])
        self.assertLess(preds[1], preds[2])
        self.assertLess(preds[2], preds[3])
        self.assertLess(preds[3], preds[4])
    
    def test_difficulty_effect_negative_a(self):
        # For negative a, the effect reverses: higher difficulty should yield higher probabilities.
        a = -1.0
        c = 0.3
        q = self.model._base_rate_to_logit(c)
        preds = self.model.predict((a, q))
        self.assertGreater(preds[0], preds[-1])
        self.assertGreater(preds[0], preds[1])
        self.assertGreater(preds[1], preds[2])
        self.assertGreater(preds[2], preds[3])
        self.assertGreater(preds[3], preds[4])

    
    def test_predict_known_values(self):
        # Test predict() with known parameter values:
        # When a=0, exp(0)=1, so p = c + (1-c)/2 for all conditions.
        a = 0.0
        c = 0.3
        expected = c + (1 - c) / 2  # 0.3 + 0.35 = 0.65
        q = self.model._base_rate_to_logit(c)
        preds = self.model.predict((a, q))
        for p in preds:
            self.assertAlmostEqual(p, expected, places=6)
    
    # === Parameter Estimation Tests ===
    def test_nll_improves_after_fit(self):
        # Test that the negative log-likelihood improves after fitting.
        a = 1.0
        c = 0.2
        q = self.model._base_rate_to_logit(c)
        init_params = [a, q]
        nll_initial = self.model.negative_log_likelihood(init_params)
        self.model.fit()
        fitted_params = [self.model.get_discrimination(), self.model.get_logit_base_rate()]
        nll_fitted = self.model.negative_log_likelihood(fitted_params)
        self.assertLess(nll_fitted, nll_initial)

    def test_high_discrimination_with_steep_data(self):
        # Create steep data with 5 conditions, 100 trials each, with accuracy rates: 0.01, 0.05, 0.50, 0.95, 0.99.
        exp_steep = Experiment()
        total_trials = 100
        accs = [0.01, 0.05, 0.50, 0.95, 0.99]
        for acc in accs:
            correct = int(round(total_trials * acc))
            incorrect = total_trials - correct
            hits = correct // 2
            correctRejections = correct - hits
            misses = 50 - hits
            falseAlarms = 50 - correctRejections
            sdt = SignalDetection(hits, misses, falseAlarms, correctRejections)
            exp_steep.add_condition(sdt)
        model_steep = SimplifiedThreePL(exp_steep)
        model_steep.fit()
        # Threshold: expect discrimination > 2.5 with the given steep data.
        self.assertGreater(model_steep.get_discrimination(), 2.5)
    
    def test_get_params_before_fit_raises(self):
        # Ensure that attempting to get parameters before fitting raises an error. Almost the same as test_get_params_before_fit.
        model = self.model  # from setUp (not fitted)
        with self.assertRaises(ValueError):
            model.get_discrimination()
        with self.assertRaises(ValueError):
            model.get_base_rate()
    
    # === Integration Test ===
    def test_integration(self):
        # Create a dataset with 5 conditions, 100 trials per condition (50 signal, 50 noise),
        # with accuracy rates exactly: 0.55, 0.60, 0.75, 0.90, 0.95.
        exp_int = Experiment()
        total_trials = 100
        accs = [0.55, 0.60, 0.75, 0.90, 0.95]
        for acc in accs:
            correct = int(round(total_trials * acc))
            incorrect = total_trials - correct
            # Assume signal and noise each contribute 50 trials. Distribute correct responses approximately equally.
            hits = correct // 2
            correctRejections = correct - hits
            misses = 50 - hits
            falseAlarms = 50 - correctRejections
            sdt = SignalDetection(hits, misses, falseAlarms, correctRejections)
            exp_int.add_condition(sdt)

        # Initialize and fit the model.
        model_int = SimplifiedThreePL(exp_int)
        model_int.fit()
        
        # Save fitted parameters for stability check.
        initial_discrimination = model_int.get_discrimination()
        initial_base_rate = model_int.get_base_rate()
        
        # Re-fit the model several times and check that the parameters remain stable.
        for _ in range(3):
            model_int.fit()
            self.assertAlmostEqual(model_int.get_discrimination(), initial_discrimination, places=3)
            self.assertAlmostEqual(model_int.get_base_rate(), initial_base_rate, places=3)
        
        # Get predictions from the model.
        predictions = model_int.predict((model_int.get_discrimination(), model_int.get_logit_base_rate()))
        
        # Compute the observed accuracy (correct rate) for each condition.
        observed = []
        for sdt in exp_int.conditions:
            obs = sdt.n_correct_responses() / sdt.n_total_responses()
            observed.append(obs)
        
        # Verify that each predicted probability is close to the corresponding observed accuracy.
        for pred, obs in zip(predictions, observed):
            self.assertAlmostEqual(pred, obs, delta=0.05)

        
    # === Corruption Tests ===  
    def test_corruption(self):
        # Test that if the user directly modifies private attributes, re-fitting recovers a consistent model.
        # First, fit the model to obtain valid parameters.
        self.model.fit()
        original_discrimination = self.model.get_discrimination()
        original_base_rate = self.model.get_base_rate()

        # Simulate accidental corruption by modifying private attributes.
        self.model._discrimination = 999
        self.model._base_rate = 999
        self.model._logit_base_rate = 999

        # Optionally, you could also modify data arrays if desired:
        # self.model._correct_array *= 0.5  # Example of a corruption; not recoverable unless re-initialized.

        # Re-fit the model to recover proper parameter estimates.
        self.model.fit()
        recovered_discrimination = self.model.get_discrimination()
        recovered_base_rate = self.model.get_base_rate()

        # Verify that the recovered parameters are not equal to the corrupted values.
        self.assertNotEqual(recovered_discrimination, 999, "Discrimination should be recovered after re-fit.")
        self.assertNotEqual(recovered_base_rate, 999, "Base rate should be recovered after re-fit.")

        # Optionally, verify that the recovered parameters are close to the original ones obtained before corruption.
        # Allowing a small delta for numerical differences.
        self.assertAlmostEqual(recovered_discrimination, original_discrimination, places=3,
                            msg="Recovered discrimination should be close to the original value.")
        self.assertAlmostEqual(recovered_base_rate, original_base_rate, places=3,
                            msg="Recovered base rate should be close to the original value.")

        # Finally, check that predictions are still in a valid range.
        predictions = self.model.predict((self.model.get_discrimination(), self.model._logit_base_rate))
        for p in predictions:
            self.assertTrue(0 <= p <= 1, "Predicted probabilities must be between 0 and 1.")


    def test_invalid_condition_count(self):
        exp_invalid = Experiment()
        # Add only 3 conditions instead of 5.
        for _ in range(3):
            sdt = SignalDetection(10, 5, 3, 7)
            exp_invalid.add_condition(sdt)
        with self.assertRaises(ValueError):
            # This should raise a ValueError because the model expects exactly 5 conditions.
            SimplifiedThreePL(exp_invalid)

    def test_inconsistent_update_of_experiment(self):
        # Create a valid Experiment with exactly 5 conditions.
        exp_valid = Experiment()
        for _ in range(5):
            sdt = SignalDetection(20, 10, 5, 15)
            exp_valid.add_condition(sdt)
        model = SimplifiedThreePL(exp_valid)
        # Record the summary based on the initial data.
        summary_initial = model.summary()

        # Now, accidentally add an extra condition to the Experiment.
        extra_sdt = SignalDetection(30, 5, 2, 18)
        exp_valid.add_condition(extra_sdt)

        # The model was already constructed; its internal data arrays should not change.
        summary_after_update = model.summary()
        # Even though the Experiment now has 6 conditions, the model's summary should reflect the original 5.
        self.assertEqual(summary_initial["n_conditions"], 5)
        self.assertEqual(summary_after_update["n_conditions"], 5)
        self.assertEqual(summary_initial, summary_after_update, "Model summary should remain unchanged after external update of Experiment.")



if __name__ == '__main__':
    unittest.main()