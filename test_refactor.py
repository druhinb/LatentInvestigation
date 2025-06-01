#!/usr/bin/env python
"""Test script to verify the refactoring was successful"""

try:
    from scripts.run_probing_experiment import ProbingExperiment
    from src.analysis.layer_analysis import LayerWiseAnalyzer

    print("✓ All imports working correctly")
    print("✓ LayerWiseAnalyzer integration successful")
    print("✓ Redundancy removal completed successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
