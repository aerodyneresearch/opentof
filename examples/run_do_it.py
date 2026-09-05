"""
OpenTof DO-IT Workflow Runner
------------------------------
Executes the full processing pipeline defined in a configuration YAML file.

Performance Note:
-----------------
If running on Windows/Intel CPUs with Hybrid Architecture (P-Cores vs E-Cores),
ensure your terminal process is not flagged in 'Efficiency Mode' by Windows 
Task Manager if processing speeds slow down unexpectedly.
"""

import argparse
import opentof as ot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenTof DO-IT Workflow Runner")
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="config_template.yaml", 
        help="Path to processing configuration YAML file"
    )
    args = parser.parse_args()
    
    ot.run_do_it(args.config)