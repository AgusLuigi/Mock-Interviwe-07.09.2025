#!/bin/bash
# Aktiviere das virtuelle Environment
source venv/bin/activate
echo "Virtuelles Environment aktiviert!"
echo "Installierte Pakete:"
pip list | grep -E "(pandas|numpy|matplotlib|seaborn|scikit-learn|plotly)"