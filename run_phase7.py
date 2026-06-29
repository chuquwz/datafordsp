# -*- coding: utf-8 -*-
"""
Phase 7: Dashboard Generation & Visualization
==============================================
Loads the segmented data, association rules, and recommendation metrics,
and generates four major research visualization figures.
"""

import os
import sys
import pandas as pd
import traceback

# Ensure src path is in sys.path
sys.path.insert(0, '.')

# Placeholder for imports

LOG_FILE = 'outputs/visualization_log.txt'
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

log = open(LOG_FILE, 'w', encoding='utf-8')

def p(msg):
    log.write(str(msg) + '\n')
    log.flush()

try:
    p("=" * 60)
    p("PHASE 7: DASHBOARD GENERATION & VISUALIZATION")
    p("=" * 60)

    # 1. Initialize Dashboard Generator
    p("\n[1/2] Initializing Dashboard Generator...")
    # Wait, need to import DashboardGenerator correctly
    from src.visualization.dashboard import DashboardGenerator
    generator = DashboardGenerator()
    p("  Dashboard generator initialized.")

    # 2. Generate Dashboard Plots
    p("\n[2/2] Generating dashboard plots...")
    generator.generate_all()
    
    # Verify saved figures
    fig_dir = 'outputs/figures'
    p(f"\n  Dashboard Visual Assets saved in {fig_dir}/:")
    for f in os.listdir(fig_dir):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(fig_dir, f))
            p(f"    - {f} ({size/1024:.1f} KB)")

    p("\n" + "=" * 60)
    p("PHASE 7 COMPLETE - Dashboard visualization successful!")
    p("=" * 60)

except Exception as e:
    p(f"\n!!! ERROR: {e}")
    p(traceback.format_exc())

finally:
    log.close()
    # Print the log output safely using utf-8 encoding
    sys.stdout.reconfigure(encoding='utf-8')
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        print(f.read())
