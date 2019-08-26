#!/bin/bash
echo "==========================================================================="
g++ -O3 -DNDEBUG -Wno-unused-result -o evaluate_depth evaluate_depth.cpp -lpng
echo "Built evaluate_depth."