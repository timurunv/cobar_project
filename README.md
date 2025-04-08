# Miniproject for BIOENG-456: Controlling behavior in animals and robots

Welcome to the Miniproject for BIOENG-456!

## Setup
`pynput` is required for the KeyBoardController in `explore_levels.py`. To install it, run:
```bash
conda activate flygym
pip install pynput
```

## Usage
To explore the levels interactively, run the `explore_levels.py` script:
```bash
conda activate flygym
python explore_levels.py --level <level> --seed <seed>
```
Replace `<level>` with the desired level number (0 to 4 for the 5 levels, -1 for just a flat terrain) and `<seed>` with the random seed for reproducibility.
