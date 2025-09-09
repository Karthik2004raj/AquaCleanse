# ETA Engine

Advanced ETA calculator with business calendars, holidays, and mixed business/calendar durations.

## Features
- Business calendars with working-hour intervals per weekday
- Holiday support via `holidays` library
- Add business or calendar duration across steps
- Example: plastic recycler → supplier pipeline

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=$(pwd) python examples/plastic_recycler_demo.py
```

## Structure
- `eta/`: core library (`calendars.py`, `business_time.py`, `models.py`, `eta_engine.py`)
- `examples/`: runnable scenarios
- `tests/`: place tests here

## Configure
Edit the example’s `build_calendar()` to set timezone, holidays, and working hours.

## Repository Setup
```bash
git init -b main
echo "# Python\n__pycache__/\n*.py[cod]\n.venv/\n.env\n" > .gitignore
git add .
git commit -m "feat: initial ETA engine with business calendar and demo"
```

